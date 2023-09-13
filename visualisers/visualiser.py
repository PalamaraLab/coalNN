import os
import sys
import datetime
import socket
import json
import yaml
import torch
import pickle
import numpy as np
import torch.nn.functional as F
import copy

from abc import ABCMeta, abstractmethod
from math import sqrt
from utils import Config, Logger, print_model_spec


class Visualiser:
    __meta_class__ = ABCMeta

    def __init__(self, options):
        self.options = options

        ##########
        # Restore session
        ##########
        self.config = None
        self.restore_session_name = ''
        if self.options.restore:
            self.restore_session()
        else:
            raise ValueError('Must specify --restore path.')

        ##########
        # Running sessions
        ##########
        self.session_name = ''
        self.initialise_session()

        self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')

        ##########
        # Data
        ##########
        self.random_pairs = None
        self.run_dataset = None
        self.run_dataloader = None
        self.input_size, self.focus_input_size, self.context_size, self.x_dim = None, None, None, None
        self.root_time = None
        self.create_data()

        if self.config.run_CoalNN:
            ##########
            # Model
            ##########
            self.model = None
            self.create_model()
            print_model_spec(self.model)
            self.model.to(self.device)
            self.global_step = 1

            ##########
            # Metrics
            ##########
            self.run_metrics = None
            self.create_metrics()

            # Restore model
            self.load_checkpoint(self.options.restore)

            ##########
            # Plotter
            ##########
            self.plotter = None
            self.create_plotter()

    @abstractmethod
    def create_data(self):
        """Create run datasets and dataloaders."""

    @abstractmethod
    def create_model(self):
        """Build the neural network."""

    @abstractmethod
    def create_plotter(self):
        """Implement the plotter."""

    @abstractmethod
    def loss_function(self):
        """Return loss function."""

    @abstractmethod
    def metric_function(self):
        """Return loss function."""

    @abstractmethod
    def create_metrics(self):
        """Implement the metrics."""

    @abstractmethod
    def forward_model(self, batch):
        """Compute the output of the model."""

    @abstractmethod
    def forward_loss(self, batch, output):
        """Compute the loss."""

    @abstractmethod
    def forward_metric(self, batch, output):
        """Compute the metric."""

    def initialise_visualisation(self):

        dataset = self.run_dataset

        random_pairs = np.random.choice(range(0, len(dataset)),
                                        size=self.config.batch_size, replace=False)
        phys_pos = dataset.simulations[0]['phys_pos']

        return random_pairs, dataset, phys_pos

    def random_batch(self, random_pairs, dataset):
        # get a batch with random pairs
        batch = {}

        for pair in random_pairs:
            batch_pair = dataset[pair]
            for key, value in batch_pair.items():
                if key in batch:
                    batch[key].append(value)
                else:
                    batch[key] = [value]

        for key in batch:
            batch[key] = torch.stack(batch[key])

        self.preprocess_batch(batch)

        return batch

    def visualise(self):
        print('Starting visualisation session..')
        print('Will decode', self.config.batch_size, 'pairs...')

        random_pairs, dataset, phys_pos = self.initialise_visualisation()
        batch = self.random_batch(random_pairs, dataset)
        with torch.no_grad():
            output = self.forward_model(batch)
            output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
        if self.config.log_tmrca:
            batch['label'] = torch.exp(batch['label'])
            output['output'] = torch.exp(output['output'])
            # batch['label'] = torch.pow(10, batch['label'])
            # output['output'] = torch.pow(10, output['output'])
        if self.config.constant_piecewise_output:
            output['output_const'] = self.get_const_piecewise(copy.deepcopy(output), self.config.const_threshold)

        loss, metric = self.random_loss_metric(batch, output)

        print("Visualisation on running set...")
        print("l2_loss", loss)
        print("l1_loss", metric)
        self.save_metrics_json('visualisation', metric, sqrt(loss))

        asmc_output = None
        if self.config.asmc_vis:
            asmc_output = self.random_asmc_output(random_pairs, dataset, batch, output)

        self.plotter.visualise('run', batch, output, phys_pos, asmc_output)

    def random_asmc_output(self, random_pairs, dataset, batch, output):
        asmc_decoder = dataset.asmc_decoder
        asmc_output_mean = []
        asmc_output_map = []
        for cpt_pair, pair in enumerate(random_pairs):
            haplotype_i, haplotype_j = dataset.simulations[0]['pair_to_haplotypes'][pair]
            map_mean = asmc_decoder.decode_pair(haplotype_i, haplotype_j)
            map_tmrca = np.asarray(map_mean[0])
            mean_tmrca = np.asarray(map_mean[1])
            random_pos_index = batch['pos_index'][cpt_pair].to('cpu')
            asmc_output_mean.append(mean_tmrca[random_pos_index])
            asmc_output_map.append(map_tmrca[random_pos_index])
        asmc_output_mean = np.asarray(asmc_output_mean).reshape(output['output'].shape)
        asmc_output_map = np.asarray(asmc_output_map).reshape(output['output'].shape)
        asmc_output = {'output_mean': torch.from_numpy(asmc_output_mean),
                       'output_map': torch.from_numpy(asmc_output_map)}
        self.preprocess_batch(asmc_output)
        return asmc_output

    @staticmethod
    def get_sample_const_piecewise(sample):
        prediction = sample[0]
        breakpoints = sample[1]
        threshold = sample[2]
        breakpoints = breakpoints[1, :]  # class 1
        # breakpoints_pos = np.arange(len(prediction))[breakpoints >= self.config.const_threshold]
        breakpoints_pos = np.arange(len(prediction))[breakpoints >= threshold]
        start_pos = 0
        for pos in breakpoints_pos:
            prediction[start_pos: pos] = prediction[start_pos: pos].mean()
            start_pos = pos
        prediction[start_pos:] = prediction[start_pos:].mean()
        return prediction

    def get_const_piecewise(self, output, threshold):
        if self.device.type == 'cuda':
            # move tensors to CPU memory first, so they can be converted to arrays
            output['output'] = output['output'].to('cpu')
            output['breakpoints'] = output['breakpoints'].to('cpu')
        threshold = output['output'].shape[0] * [threshold]
        sample = zip(output['output'], output['breakpoints'], threshold)
        const_piecewise_output = list(map(self.get_sample_const_piecewise, sample))
        return torch.stack(const_piecewise_output).to(self.device)

    @staticmethod
    def l2_loss(label, output):
        return torch.mean((output - label) ** 2)

    @staticmethod
    def l1_loss(label, output):
        return F.l1_loss(output, label, reduction='mean')

    def random_loss_metric(self, batch, output):
        loss = int(torch.sqrt(self.l2_loss(batch['label'], output['output'])).item())
        metric = int(self.l1_loss(batch['label'], output['output']).item())
        return loss, metric

    def run_prediction(self, batch):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        return output

    def save_metrics_json(self, name, l1, l2):
        filename = os.path.join(self.session_name, name + '_metrics.json')
        output = {'l1': float("{:.3f}".format(l1)), 'l2': float("{:.3f}".format(l2))}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def load_checkpoint(self, path):
        checkpoint_name = os.path.join(path, 'checkpoint')
        map_location = 'cuda' if self.config.gpu else 'cpu'
        checkpoint = torch.load(checkpoint_name, map_location=map_location)

        self.model.load_state_dict(checkpoint['model'])
        print('Loaded model weights from {}\n'.format(checkpoint_name))

    def preprocess_batch(self, batch):
        if self.device.type == 'cuda':
            # Cast to device
            for key, value in batch.items():
                batch[key] = value.to(self.device)

    def restore_session(self):
        config_path = os.path.join(self.options.restore, 'config.yml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

        # Additional parameters for running
        config_run_file = os.path.join(self.options.config)
        with open(config_run_file) as f:
            config_run = yaml.safe_load(f)

        if self.options.seed:
            config_run['seed_run'] = int(self.options.seed)

        if self.options.size:
            config_run['sample_size_run'] = int(self.options.size)
            # config_run['sample_size'] = int(self.options.size)

        if self.options.ref_size:
            config_run['val_ref_size'] = int(self.options.ref_size)

        if self.options.const_threshold:
            config_run['constant_piecewise_output'] = True
            config_run['const_threshold'] = float(self.options.const_threshold)

        if self.options.downsample_size:
            config_run['downsample_size'] = int(self.options.downsample_size)

        self.config.update_config(config_run)
        self.restore_session_name = self.config.session_name

        # Compare git hash
        # current_git_hash = get_git_hash()
        # with open(os.path.join(self.restore_session_name, 'git_hash')) as f:
        #    previous_git_hash = f.read().splitlines()[0]
        # if current_git_hash != previous_git_hash:
        #    print('Restoring model with a different git hash.')
        #    print(f'Previous: {previous_git_hash}')
        #    print(f'Current: {current_git_hash}\n')

    def create_dataset_name(self):

        if 'demography' in self.config:
            if "CEU" in str(self.config.demography):
                demo = 'CEU'
            elif "constant" in str(self.config.demography):
                demo = 'constant'
            else:
                raise ValueError('Demographic model is unknown.')
        else:
            demo = 'constant'

        recombination = ''
        if 'rec_rate' in self.config:
            recombination = '.rec_const.' + str(self.config.rec_rate) \
                            + '.length.' + str(self.config.length)

        if self.config.data_type == 'array':
            mode = '.array'
        elif self.config.data_type == 'sequence':
            mode = '.sequence'
        elif self.config.data_type == 'impute':
            mode = '.impute.ref_size.' + str(self.config.val_ref_size)

        const_threshold = ''
        if 'const_threshold' in self.config:
            const_threshold = '.const_threshold.' + str(self.config.const_threshold)

        dataset_name = 'CHR.' + str(self.config.chr) \
                       + '.Ne.' + str(self.config.Ne) \
                       + '.S_run.' + str(self.config.sample_size_run) \
                       + mode \
                       + '.DEMO.' + demo \
                       + recombination \
                       + '.downsample_size.' + str(self.config.downsample_size) \
                       + const_threshold \
                       + '.bin.' + str(self.config.model.bin) + str(self.config.model.bin_unit) \
                       + '.seed.' + str(self.config.seed_run)

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
            self.config.tag,
            dataset_name)
        session_name = os.path.join(self.config.output_path, session_name)
        os.makedirs(session_name)
        return session_name

    def initialise_session(self):
        self.session_name = self.create_session_name()
        self.config.session_name = self.session_name
        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))
        # Copy config run file to output_path
        config_run_file = os.path.join(self.options.config)
        with open(config_run_file) as f:
            config_run = yaml.safe_load(f)
        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(config_run, f)
