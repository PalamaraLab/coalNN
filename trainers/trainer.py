import os
import sys
import datetime
import socket
from abc import ABCMeta, abstractmethod
from time import time

import yaml
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

from utils import Config, Logger, format_time, print_model_spec


class Trainer:
    __meta_class__ = ABCMeta

    def __init__(self, options):
        self.options = options

        ##########
        # Trainer utils
        ##########
        self.global_step = 0
        self.start_time = None

        ##########
        # Initialise/restore session
        ##########
        self.config = None
        self.session_name = ''
        if self.options.config:
            self.initialise_session()
        elif self.options.restore:
            self.restore_session()
        else:
            raise ValueError('Must specify --config or --restore path.')

        self.tensorboard = SummaryWriter(self.session_name, comment=self.config.tag)
        self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')

        ##########
        # Data
        ##########
        self.train_dataset, self.val_dataset = None, None
        self.train_dataloader, self.val_dataloader = None, None
        self.input_size, self.focus_input_size, self.context_size, self.x_dim = None, None, None, None
        self.create_data()

        ##########
        # Model
        ##########
        self.model = None
        self.create_model()
        print_model_spec(self.model)
        self.model.to(self.device)

        ##########
        # Loss
        ##########
        self.multi_task_loss = None
        self.create_multi_task_loss()

        ##########
        # Optimiser
        ##########
        self.optimiser = None
        self.create_optimiser()

        ##########
        # Metrics
        ##########
        self.train_metrics = None
        self.val_metrics = None
        self.create_metrics()

        ##########
        # Plotter
        ##########
        self.plotter = None
        # self.create_plotter()

        # Restore model
        if self.options.restore:
            self.load_checkpoint(self.options.restore)

    @abstractmethod
    def create_data(self):
        """Create train/val datasets and dataloaders."""

    @abstractmethod
    def create_model(self):
        """Build the neural network."""

    @abstractmethod
    def create_multi_task_loss(self):
        """Build the multi task loss wrapper."""

    @abstractmethod
    def create_optimiser(self):
        """Create the model's optimiser."""

    @abstractmethod
    def loss_function(self, batch, output):
        """Return loss function."""

    @abstractmethod
    def metric_function(self, batch, output, mode='train'):
        """Return loss function."""

    @abstractmethod
    def create_metrics(self):
        """Implement the metrics."""

    @abstractmethod
    def create_plotter(self):
        """Implement the plotter."""

    @abstractmethod
    def visualise(self, mode):
        """Visualise prediction on tensorboard."""

    @abstractmethod
    def forward_model(self, batch):
        """Compute the output of the model."""

    @abstractmethod
    def forward_loss(self, batch, output):
        """Compute the loss."""

    @abstractmethod
    def forward_metric(self, batch, output):
        """Compute the metric."""

    @abstractmethod
    def simulate_training_set(self):
        """Simulate a random training set."""

    def train(self):
        print('Starting training session..')
        self.model.train()

        self.start_time = time()
        while self.global_step < self.config.n_epochs:

            if self.global_step % self.config.n_simulate_training_set == 0:
                # if self.global_step == self.config.n_epochs_convergence and not self.config.threading_sampler:
                if self.config.do_threading_sampler:
                    if self.global_step % 2 == 0:
                        self.config.threading_sampler = False
                        self.config.update_config_file(self.session_name, 'threading_sampler', False)
                        # self.val_metrics.update(float('inf'), self.global_step)
                    else:
                        self.config.threading_sampler = True
                        self.config.update_config_file(self.session_name, 'threading_sampler', True)
                self.simulate_training_set()

            self.global_step += 1

            self.train_epoch()

            if self.global_step % self.config.val_epochs == 0:
                self.val()

            # Visualise
            if self.global_step % self.config.vis_epochs == 0:
                self.visualise('train')
                self.visualise('val')

            self.update_learning_rate()

            if self.early_stopping():
                print('Best train score: {:.3f}'.format(self.train_metrics.value))
                print('Best val score: {:.3f}'.format(self.val_metrics.value))
                break

            print('\n')

            self.tensorboard.flush()

        self.tensorboard.close()

    def train_epoch(self):
        print('-' * 100)
        print('Training')
        print('-' * 100)

        epoch_loss = 0
        epoch_score = 0
        print_iteration_loss = 0
        data_fetch_time = 0
        model_update_time = 0
        t0 = time()

        fraction = 1

        for iteration, batch in tqdm(enumerate(self.train_dataloader, 1), total=int(fraction * len(self.train_dataloader))):
            score, loss, fetch_time, update_time = self.train_step(batch)
            epoch_loss += loss
            epoch_score += score
            print_iteration_loss += loss
            data_fetch_time += fetch_time
            model_update_time += update_time
            if iteration % self.config.print_iterations == 0:
                self.print_iteration(iteration, print_iteration_loss)
                print_iteration_loss = 0
            if iteration == int(fraction * len(self.train_dataloader)):
                break

        epoch_loss /= int(fraction * len(self.train_dataloader))
        epoch_score /= int(fraction * len(self.train_dataloader))

        # Print
        step_duration = time() - t0
        self.print_log(epoch_loss, step_duration, data_fetch_time, model_update_time)
        self.tensorboard.add_scalar('train/loss', epoch_loss, self.global_step)
        self.train_metrics.evaluate(epoch_score, self.global_step)
        print('-' * 100)

    def train_step(self, batch):
        # Fetch data
        t0 = time()
        self.preprocess_batch(batch)
        fetch_time = time() - t0

        # Forward pass
        t1 = time()
        if self.multi_task_loss:
            loss, output, log_vars = self.multi_task_loss(batch)
        else:
            output = self.forward_model(batch)
            loss = self.forward_loss(batch, output)
        score = self.forward_metric(batch, output)

        # Backward pass
        self.optimiser.zero_grad()
        loss.backward()

        # if self.config.clip_grad:
        #    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip_grad)

        self.optimiser.step()
        update_time = time() - t1

        return score.item(), loss.item(), fetch_time, update_time

    def val(self):
        print('-' * 100)
        print('Validation')
        print('-' * 100)
        self.model.eval()

        fraction = 0.05

        with torch.no_grad():
            val_loss = 0
            val_score = 0
            for iteration, batch in tqdm(enumerate(self.val_dataloader, 1), total=int(fraction * len(self.val_dataloader))):
                score, loss = self.val_step(batch)
                val_score += score
                val_loss += loss
                if iteration == int(fraction * len(self.val_dataloader)):
                    break

            val_loss /= int(fraction * len(self.val_dataloader))
            val_score /= int(fraction * len(self.val_dataloader))
            print(f'Val loss: {val_loss:.4f}')
            print('Val score: {:.3f}'.format(val_score))
            self.tensorboard.add_scalar('val/loss', val_loss, self.global_step)
            if self.val_metrics.evaluate(val_score, self.global_step):
                self.save_checkpoint()
            else:
                self.save_checkpoint(file_name='last_checkpoint')
            print('-' * 100)

        self.model.train()

    def val_step(self, batch):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        loss = self.forward_loss(batch, output)
        score = self.forward_metric(batch, output, mode='val')

        return score.item(), loss.item()

    def update_learning_rate(self):
        """Decide whether or not to update learning rate."""
        if self.global_step % self.config.n_learning_rate_update == 0:
            print('Network has not updated learning rate since {} epochs.\n Updating learning rate.'
                  .format(self.config.n_learning_rate_update))
            for param_group in self.optimiser.param_groups:
                param_group['lr'] /= self.config.learning_rate_update

    def early_stopping(self):
        """Decide whether or not to stop training."""
        if (self.global_step - self.val_metrics.n_step) >= self.config.n_early_stopping:
            print('Training stopped as network has not improved on validation set since {} epochs.'
                  .format(self.config.n_early_stopping))
            return True
        else:
            return False

    def print_log(self, loss, step_duration, data_fetch_time, model_update_time):
        """Print a log statement to the terminal."""
        samples_per_sec = self.config.batch_size / step_duration
        time_so_far = time() - self.start_time
        training_time_left = (self.config.n_epochs / self.global_step - 1.0) * time_so_far
        print_string = 'Epoch {:>6}/{} | examples/s: {:5.1f}' + \
                       ' | loss: {:.4f} | time elapsed: {} | time left: {}'
        print(print_string.format(self.global_step, self.config.n_epochs, samples_per_sec,
                                  loss, format_time(time_so_far), format_time(training_time_left)))
        print('Fetch data time: {:.0f}ms, model update time: {:.0f}ms'.format(1000 * data_fetch_time,
                                                                              1000 * model_update_time))

    def print_iteration(self, iteration, loss):
        """Print a log statement to the terminal."""
        print_string = 'Epoch {:>6}/{} | iterations: {} | loss: {:.4f}'
        tqdm.write(print_string.format(self.global_step, self.config.n_epochs,
                                       iteration, loss / self.config.print_iterations))

    def save_checkpoint(self, file_name='checkpoint'):
        checkpoint = dict(model=self.model.state_dict(),
                          optimiser=self.optimiser.state_dict(),
                          global_step=self.global_step
                          )

        checkpoint_name = os.path.join(self.session_name, file_name)
        torch.save(checkpoint, checkpoint_name)
        print('Model saved to: {}'.format(checkpoint_name))

    def load_checkpoint(self, path):

        checkpoint_name = os.path.join(path, 'last_checkpoint')

        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(path, 'checkpoint')

        copyfile(checkpoint_name, os.path.join(self.session_name, 'checkpoint'))
        map_location = 'cuda' if self.config.gpu else 'cpu'
        checkpoint = torch.load(checkpoint_name, map_location=map_location)

        self.model.load_state_dict(checkpoint['model'])
        if self.config.resume_training:
            self.optimiser.load_state_dict(checkpoint['optimiser'])
            self.global_step = checkpoint['global_step']
            self.train_metrics.load_json(path)
            self.val_metrics.load_json(path)
            self.train_metrics.save_json()
            self.val_metrics.save_json()
            print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))
        else:
            print('Loaded model weights from {}\n'.format(checkpoint_name))

    def preprocess_batch(self, batch):
        # Cast to device
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def initialise_session(self):
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if self.options.demography:
            self.config['demography'] = self.options.demography

        if self.options.output_path:
            self.config['output_path'] = self.options.output_path

        self.session_name = self.create_session_name()
        self.config['session_name'] = self.session_name

        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f)
        self.config = Config(self.config)

        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))

    def restore_session(self):
        config_path = os.path.join(self.options.restore, 'config.yml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.session_name = self.create_session_name()
        self.config['session_name'] = self.session_name
        self.config['resume_training'] = True

        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f)
        self.config = Config(self.config)

        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))

    def create_dataset_name(self):

        populations = ["CEU.Terhorst", "ACB", "CEU", "ESN", "GWD", "KHV", "PEL", "STU", "ASW", "CHB", "FIN", "IBS",
                       "LWK", "PJL", "TSI", "BEB", "CHS", "GBR", "ITU", "MSL", "PUR", "YRI", "CDX",
                       "CLM", "GIH", "JPT", "MXL"]
        if "constant" in self.config['demography']:
            demo = 'constant' + '.Ne.' + str(self.config['Ne'])
        elif self.config['demography'] in populations:
            demo = str(self.config['demography'])
        else:
            raise ValueError('Demographic model is unknown.')

        recombination = ''
        if 'rec_rate' in self.config:
            recombination = '.rec_const.' + str(self.config['rec_rate']) \
                            + '.length.' + str(self.config['length'])

        if self.config['data_type'] == 'array':
            mode = '.array'
        elif self.config['data_type'] == 'sequence':
            mode = '.sequence'
        elif self.config['data_type'] == 'impute':
            mode = '.impute.ref_size.' + str(self.config['val_ref_size'])

        log = ''
        if self.config['log_tmrca']:
            log = '.log'

        offset = ''
        if 'use_offset' in self.config and self.config['use_offset']:
            offset = '.offset'
        
        dataset_name = '.S_train.' + str(self.config['sample_size_train']) \
                       + '.S_val.' + str(self.config['sample_size_val']) \
                       + mode \
                       + '.DEMO.' + demo \
                       + offset \
                       + recombination \
                       + log \
                       + '.downsample_size.' + str(self.config['downsample_size']) \
                       + '.bin.' + str(self.config['model']['bin']) + str(self.config['model']['bin_unit']) \
                       + '.seed_val.' + str(self.config['seed_val'])

        return dataset_name

    def create_session_name(self):

        dataset_name = self.create_dataset_name()
        now = datetime.datetime.now()
        session_name = '{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_session_{}_{}_{}'.format(
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            socket.gethostname(),
            self.config['tag'],
            dataset_name)
        session_name = os.path.join(self.config['output_path'], session_name)
        os.makedirs(session_name)
        return session_name
