import torch
import os

from torch.utils.data import DataLoader
from visualisers.visualiser import Visualiser
from shutil import copyfile
from data.genome import GenomeDataset
from data.sampler import Subset
from models.cnn_network import CnnNetwork
from metrics.accuracy import AccuracyMetrics
from plotter.plotter import Plotter


class MsprimeVisualiser(Visualiser):
    def create_data(self):

        print('-' * 100)
        print('Building dataset...')

        self._get_attributes()

        self.run_dataset = GenomeDataset(self.config, mode='run',
                                         input_size=self.input_size,
                                         focus_input_size=self.focus_input_size,
                                         context_size=self.context_size,
                                         root_time=self.root_time)
        print('Dataset has {} pairs of haplotypes.\n'.format(len(self.run_dataset)))

        num_haplotypes = 2 * self.config.sample_size
        num_diploids = self.config.sample_size
        num_pairs = int(num_diploids * (num_haplotypes - 2))
        all_pairs = [pair
                     for pair in self.run_dataset.simulations[0]['haplotypes_to_pair']]
        # self.random_pairs = random.sample(all_pairs, k=num_pairs)
        self.random_pairs = all_pairs[:num_pairs]  # removing randomness for debugging

        if self.config.run_CoalNN:
            sampler = Subset(self.random_pairs, self.run_dataset.simulations[0]['haplotypes_to_pair'])
            self.run_dataloader = DataLoader(self.run_dataset, self.config.batch_size,
                                             num_workers=self.config.n_workers, sampler=sampler)

    def _get_attributes(self):
        self.x_dim = self.config.x_dim
        self.x_dim[0] = self.config.batch_size
        self.input_size = self.x_dim[-1]
        self.focus_input_size = self.config.focus_input_size
        self.context_size = self.config.context_size
        self.root_time = self.config.root_time

    def create_model(self):
        print('-' * 100)
        print('Building model {}...'.format(self.config.model.network))
        print('Batches have shape {}.'.format(self.x_dim))
        if self.config.model.network == 'cnn':
            if self.config.model.restriction_activation:
                clamp = False
            else:
                clamp = True
            self.model = CnnNetwork(self.focus_input_size, self.x_dim, self.config, self.root_time, clamp)

    def create_plotter(self):
        self.plotter = Plotter(self.config, vis=True)

    def loss_function(self, label, output):
        return torch.sum((output - label) ** 2)

    def metric_function(self, label, output):
        return torch.sum(torch.abs((output - label)))

    def create_metrics(self):
        self.run_metrics = AccuracyMetrics('run', self.session_name)

    def forward_model(self, batch):
        return self.model(batch)

    def forward_loss(self, label, output):
        return self.loss_function(label, output)

    def forward_metric(self, label, output):
        return self.metric_function(label, output)

