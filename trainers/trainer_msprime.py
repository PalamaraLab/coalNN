from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy
import random
import os
import math

from trainers.trainer import Trainer
from data.genome import GenomeDataset
from models.cnn_network import CnnNetwork
from metrics.accuracy import AccuracyMetrics
from plotter.plotter import Plotter
from models.multi_task_loss import MultiTaskLossWrapper
from shutil import copyfile
from data.sampler import ValidationThreadingSampler
from utils import load_numpy


class MsprimeTrainer(Trainer):
    def create_data(self):

        print('-' * 100)
        print('Building validation dataset...')
        self.val_dataset = GenomeDataset(self.config, mode='val')

        if self.config.val_threading_sampler:
            if self.options.val_ordered_pairs and \
                    os.path.exists(os.path.join(self.options.val_ordered_pairs, 'val_ordered_pairs.npy')):
                print('Loading ordered pairs for validation from {}...'.format(self.options.val_ordered_pairs))
                pairs = load_numpy(self.options.val_ordered_pairs, 'val_ordered_pairs')
            elif self.options.restore and \
                    os.path.exists(os.path.join(self.options.restore, 'val_ordered_pairs.npy')) and \
                    self.config.resume_training:
                print('Loading ordered pairs for validation from {}...'.format(self.options.restore))
                pairs = load_numpy(self.options.restore, 'val_ordered_pairs')
            else:
                pairs = None
            print('Building model validation threading sampler...')
            sampler = ValidationThreadingSampler(self.val_dataset.simulations,
                                                 self.session_name,
                                                 pairs=pairs)

        else:
            sampler = None

        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size,
                                         num_workers=self.config.n_workers, shuffle=False, sampler=sampler)
        print('Validation dataset has {} pairs of haplotypes.\n'.format(len(self.val_dataset)))

        self.x_dim = self._get_x_dim()
        self.input_size = self.val_dataset.input_size
        self.focus_input_size = self.val_dataset.focus_input_size
        self.context_size = self.val_dataset.context_size
        self.config.update_config_file(self.session_name, 'x_dim', self.x_dim)
        self.config.update_config_file(self.session_name, 'focus_input_size', self.val_dataset.focus_input_size)
        self.config.update_config_file(self.session_name, 'context_size', self.val_dataset.context_size)
        self.config.update_config_file(self.session_name, 'root_time', self.val_dataset.root_time)

    def _get_x_dim(self):
        return [self.config.batch_size, self.val_dataset.num_features, self.val_dataset.input_size]

    def create_model(self):
        print('-' * 100)
        print('Building model {}...'.format(self.config.model.network))
        print('Batches have shape {}.'.format(self.x_dim))
        if self.config.model.network == 'cnn':
            self.model = CnnNetwork(self.focus_input_size, self.x_dim, self.config, self.val_dataset.root_time)
            copyfile('./models/cnn_network.py', os.path.join(self.session_name, 'model.py'))

    def create_multi_task_loss(self):
        if self.config.multi_task_loss:
            print('-' * 100)
            print('Building multi task loss wrapper...')
            self.multi_task_loss = MultiTaskLossWrapper(2, self.model, self.config)
            copyfile('./models/multi_task_loss.py', os.path.join(self.session_name, 'multi_task_loss.py'))

    def simulate_training_set(self):
        print('-' * 100)
        print('Building training dataset ...')
        self.train_dataset = GenomeDataset(self.config, mode='train',
                                           input_size=self.val_dataset.input_size,
                                           focus_input_size=self.val_dataset.focus_input_size,
                                           context_size=self.val_dataset.context_size,
                                           root_time=self.val_dataset.root_time)
        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size,
                                           num_workers=self.config.n_workers, shuffle=False)

        print('Training dataset has {} pairs of haplotypes.\n'.format(len(self.train_dataset)))

    @staticmethod
    def l2_loss(label, output):
        return torch.mean((output - label) ** 2)

    @staticmethod
    def l1_loss(label, output):
        return F.l1_loss(output, label, reduction='mean')

    def regression_weight(self, label):
        if self.config.log_tmrca:
            data = torch.exp(label)
        data = torch.log10(data)
        data = torch.max(data, torch.zeros_like(data))
        min_value = data.min().int()
        max_value = data.max().int() + 1
        weights = torch.histc(data, min=min_value, max=max_value, bins=max_value - min_value)
        weights = 1. / weights
        bins_allocations = data.long() - min_value
        data_weights = weights[bins_allocations]
        data_weights = torch.nn.functional.normalize(data_weights, p=1, dim=-1)
        return data_weights

    def weighted_l1_loss(self, label, output):
        weight = self.regression_weight(label)
        l1_loss = weight * torch.abs(label - output)
        return torch.mean(torch.sum(l1_loss, dim=-1))

    @staticmethod
    def huber_loss(label, output):
        return F.smooth_l1_loss(output, label, reduction='mean')

    @staticmethod
    def cross_entropy_weight(data):
        unique, counts = torch.unique(data, sorted=True, return_counts=True)
        counts = (1. / torch.sum(counts).item()) * counts
        weight = counts.pow_(-1)
        weight = (1. / torch.sum(weight)) * weight
        return weight

    @staticmethod
    def cross_entropy_with_logits(output, label):
        weight = MsprimeTrainer.cross_entropy_weight(label)
        return F.cross_entropy(output, label, reduction='mean', weight=weight)

    def loss_function(self, batch, output):
        huber_loss = self.huber_loss(batch['label'], output['output'])
        breakpoints_loss = self.cross_entropy_with_logits(output['breakpoints'], batch['breakpoints'])
        return huber_loss + breakpoints_loss

    def metric_function(self, batch, output, mode='train'):
        if mode == 'train':
            return self.l1_loss(batch['label'], output['output'])
        elif mode == 'val':
            return self.weighted_l1_loss(batch['label'], output['output'])

    def create_optimiser(self):
        if self.multi_task_loss:
            parameters_with_grad = filter(lambda p: p.requires_grad, self.multi_task_loss.parameters())
        else:
            parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad,
                              lr=self.config.learning_rate,
                              weight_decay=float(self.config.weight_decay))

    def create_metrics(self):
        self.train_metrics = AccuracyMetrics('train', self.session_name, self.tensorboard)
        self.val_metrics = AccuracyMetrics('val', self.session_name, self.tensorboard)

    def create_plotter(self):
        self.plotter = Plotter(self.config, tensorboard=self.tensorboard)

    def forward_model(self, batch):
        return self.model(batch)

    def forward_loss(self, batch, output):
        return self.loss_function(batch, output)

    def forward_metric(self, batch, output, mode='train'):
        return self.metric_function(batch, output, mode)

    @staticmethod
    def get_sample_const_piecewise(sample):
        prediction = sample[0]
        breakpoints = sample[1]
        const_threshold = sample[2]
        breakpoints = breakpoints[1, :]  # class 1
        breakpoints_pos = np.arange(len(prediction))[breakpoints >= const_threshold]
        start_pos = 0
        for pos in breakpoints_pos:
            prediction[start_pos: pos] = prediction[start_pos: pos].mean()
            start_pos = pos
        prediction[start_pos:] = prediction[start_pos:].mean()
        return prediction

    def get_const_piecewise(self, output, mode):
        if self.device.type == 'cuda':
            # move tensors to CPU memory first, so they can be converted to arrays
            output['output'] = output['output'].to('cpu')
            output['breakpoints'] = output['breakpoints'].to('cpu')
        const_threshold = random.random()
        const_threshold_iter = self.config.batch_size * [const_threshold]
        sample = zip(output['output'], output['breakpoints'], const_threshold_iter)
        self.tensorboard.add_scalar(mode + '/vis_const_threshold', const_threshold, self.global_step)
        const_piecewise_output = list(map(self.get_sample_const_piecewise, sample))
        return torch.stack(const_piecewise_output), const_threshold

    def visualise(self, mode):

        random_pairs, dataset, phys_pos = self.initialise_visualisation(mode)
        batch = self.random_batch(random_pairs, dataset)
        with torch.no_grad():
            if self.multi_task_loss:
                log_vars = self.multi_task_loss.log_vars.data.tolist()
            output = self.forward_model(batch)
            # output['breakpoints'] = torch.sigmoid(output['breakpoints'])
            output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
        if self.config.log_tmrca:
            batch['label'] = torch.exp(batch['label'])
            output['output'] = torch.exp(output['output'])
            # batch['label'] = torch.pow(10, batch['label'])
            # output['output'] = torch.pow(10, output['output'])
        if self.config.const_piecewise:
            output['output_const'], const_threshold = self.get_const_piecewise(copy.deepcopy(output), mode)
        self.preprocess_batch(output)

        loss, metric = self.random_loss_metric(mode, batch, output)
        self.tensorboard.add_scalars(mode + '/vis_l2_loss', loss, self.global_step)
        self.tensorboard.add_scalars(mode + '/vis_l1_loss', metric, self.global_step)

        if mode == 'train':
            print("Visualisation on training set...")
            print("l2_loss", loss)
            print("l1_loss", metric)
            if self.multi_task_loss:
                variances = [math.exp(log_var) for log_var in log_vars]
                print("Multi task loss variances: " + ' '.join(f"{variance:.3f}" for variance in variances))
                variances_dict = {'sigma_1': variances[0],
                                  'sigma_2': variances[1]}
                self.tensorboard.add_scalars(mode + '/vis_variances', variances_dict, self.global_step)

        elif mode == 'val':
            print("Visualisation on validation set...")
            print("l2_loss", loss)
            print("l1_loss", metric)

        if self.config.const_piecewise:
            print("Constant threshold: {:.3f}".format(const_threshold))

    def random_loss_metric(self, mode, batch, output, asmc_output=None):

        loss = {'DL': int(torch.sqrt(self.l2_loss(batch['label'], output['output'])).item())}
        metric = {'DL': int(self.l1_loss(batch['label'], output['output']).item())}
        if self.config.const_piecewise:
            loss['DL_const'] = int(torch.sqrt(self.l2_loss(batch['label'], output['output_const'])).item())
            metric['DL_const'] = int(self.l1_loss(batch['label'], output['output_const']).item())

        return loss, metric

    def initialise_visualisation(self, mode):

        if mode == 'train':
            dataset = self.train_dataset
        elif mode == 'val':
            dataset = self.val_dataset

        random_pairs = np.random.choice(range(0, len(dataset)),
                                        size=self.config.batch_size, replace=False)
        phys_pos = dataset.simulations[0]['phys_pos']

        return random_pairs, dataset, phys_pos

    def random_batch(self, random_pairs, dataset):
        # get a batch with random pairs
        batch = {'label': [],
                 'pos_index': [],
                 'breakpoints': [],
                 'input': []}

        for pair in random_pairs:
            batch_pair = dataset[pair]
            for key, value in batch_pair.items():
                batch[key].append(value)

        batch['input'] = torch.stack(batch['input'])
        batch['label'] = torch.stack(batch['label'])
        batch['pos_index'] = torch.stack(batch['pos_index'])
        batch['breakpoints'] = torch.stack(batch['breakpoints'])
        self.preprocess_batch(batch)

        return batch
