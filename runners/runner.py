"""
Code adapted from Anthony Hu.
https://github.com/anthonyhu/ml-research
"""

import os
import sys
from abc import ABCMeta, abstractmethod

import yaml
import torch
import copy
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from math import sqrt
import datetime
import socket
import json
import random
import math
import pdb

from utils import Config, Logger, print_model_spec, track, save_numpy, save_torch, plot_TMRCA_pred_histogram


class Runner:
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
        self.const_thresholds = None
        self.run_dataset = None
        self.run_dataloader = None
        self.input_size, self.focus_input_size, self.context_size, self.x_dim = None, None, None, None
        self.root_time = None

        self.run_CoalNN = hasattr(self.config, 'run_CoalNN') and self.config.run_CoalNN
        self.visualise_first_layer = hasattr(self.config, 'visualise_first_layer') and self.config.visualise_first_layer
        self.saliency_map = hasattr(self.config, 'saliency_map') and self.config.saliency_map
        self.perturb_maf = hasattr(self.config, 'perturb_maf') and self.config.perturb_maf

        self.create_data()

        if self.run_CoalNN or self.visualise_first_layer or self.saliency_map or self.perturb_maf:
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

        if self.config.scatter_plot:
            self.TMRCA_LABEL = []
            self.TMRCA_PREDICTION = []
            self.TMRCA_PREDICTION_CONST = []
            self.MAF = []

        if hasattr(self.config, 'const_threshold'):
            self.const_thresholds = [self.config.const_threshold]
        else:
            self.const_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

    @abstractmethod
    def create_data(self):
        """Create run datasets and dataloaders."""

    @abstractmethod
    def create_model(self):
        """Build the neural network."""

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

    def run(self):
        print('Running session..')

        if self.perturb_maf:
            self.run_CoalNN_perturb_maf()

        if self.visualise_first_layer:
            self.compute_correlation_xor()

        if self.saliency_map:
            self.compute_saliency_map_per_site() 

        if self.run_CoalNN:
            self.run_CoalNN()

    def compute_correlation_xor(self):
        print('Starting CoalNN session to compute correlation between first hidden layer output and XOR..')
        print('Will decode', len(self.random_pairs), 'pairs...')

        assert self.config.data_type == 'impute', 'Requires imputed data...'

        self.model.eval()
        num_variants = self.run_dataset.simulations[0]['num_variants']
        n_channels = self.config.model.h_dim[0]
        correlation_per_channel_xor = torch.zeros(n_channels, device=self.device)
        correlation_per_channel_and = torch.zeros(n_channels, device=self.device)
        n_correlations_xor = n_channels * [0]
        n_correlations_and = n_channels * [0]
        start_pos_range = range(0, num_variants + 2 * self.context_size - self.input_size, self.focus_input_size)

        fraction = 1

        n = self.focus_input_size

        with torch.no_grad():

            for iteration, batch in tqdm(enumerate(self.run_dataloader),
                                         total=len(self.run_dataloader),
                                         desc='Batch',
                                         position=0):

                for pos_iteration, start_pos in tqdm(enumerate(start_pos_range),
                                                     total=len(start_pos_range),
                                                     desc='Position',
                                                     position=1):

                    # make input
                    sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                    start_label = pos_iteration * self.focus_input_size
                    sub_batch['label'] = batch['label'][:, start_label: start_label + self.focus_input_size]
                    sub_batch['breakpoints'] = batch['breakpoints'][:, start_label: start_label + self.focus_input_size]
                    sub_batch['pos_index'] = torch.from_numpy(np.asarray(
                        range(start_label, start_label + self.focus_input_size)))
                    batch_size = sub_batch['input'].shape[0]

                    # compute prediction
                    self.preprocess_batch(sub_batch)
                    output = self.forward_model(sub_batch)
                    hidden_layer_1_output = output['hidden_layer_1_output']

                    # compute correlation
                    and_input = torch.logical_and(sub_batch['input'][:, 0, :], sub_batch['input'][:, 1, :])
                    and_input = and_input[:, self.context_size:self.focus_input_size + self.context_size]
                    xor_input = torch.logical_xor(sub_batch['input'][:, 0, :], sub_batch['input'][:, 1, :])
                    xor_input = xor_input[:, self.context_size:self.focus_input_size + self.context_size]

                    dilation = math.ceil(self.focus_input_size / (2 * self.config.model.kernel_size[0]))
                    hidden_layer_1_output_size = (self.input_size - dilation * (
                            self.config.model.kernel_size[0] - 1) - 1) + 1
                    assert (hidden_layer_1_output_size - self.focus_input_size) % 2 == 0
                    hidden_layer_1_output_context_size = int((hidden_layer_1_output_size - self.focus_input_size) / 2)
                    hidden_layer_1_output = hidden_layer_1_output[:, :,
                                            hidden_layer_1_output_context_size:
                                            self.focus_input_size + hidden_layer_1_output_context_size]

                    for channel in range(n_channels):
                        for batch_sample in range(batch_size):
                            x = hidden_layer_1_output[batch_sample, channel, :]
                            y = xor_input[batch_sample, :]
                            z = and_input[batch_sample, :]
                            sum_xy = torch.sum(x * y)
                            sum_xz = torch.sum(x * z)
                            sum_x = x.sum()
                            sum_y = y.sum()
                            sum_z = z.sum()
                            sum_x_2 = (x ** 2).sum()
                            sum_y_2 = (y ** 2).sum()
                            sum_z_2 = (z ** 2).sum()
                            r_xor = (n * sum_xy - sum_x * sum_y) / (
                                torch.sqrt((n * sum_x_2 - sum_x ** 2) * (n * sum_y_2 - sum_y ** 2)))
                            r_and = (n * sum_xz - sum_x * sum_z) / (
                                torch.sqrt((n * sum_x_2 - sum_x ** 2) * (n * sum_z_2 - sum_z ** 2)))
                            if r_xor == r_xor:
                                # r is not nan
                                correlation_per_channel_xor[channel] += r_xor
                                # pdb.set_trace()
                                n_correlations_xor[channel] += 1
                            if r_and == r_and:
                                # r is not nan
                                correlation_per_channel_and[channel] += r_and
                                n_correlations_and[channel] += 1

                if iteration == int(fraction * len(self.run_dataloader)):
                    break

        for channel in range(n_channels):
            correlation_per_channel_xor[channel] /= max(n_correlations_xor[channel], 1)
            print('Channel', channel,
                  'Pearson correlation with XOR', correlation_per_channel_xor[channel])
            correlation_per_channel_and[channel] /= max(n_correlations_and[channel], 1)
            print('Channel', channel,
                  'Pearson correlation with AND', correlation_per_channel_and[channel])

        self.save_correlations_json(correlation_per_channel_xor, 'xor_')
        self.save_correlations_json(correlation_per_channel_and, 'and_')

    def save_correlations_json(self, correlations, name):
        filename = os.path.join(self.session_name, name + 'correlations.json')
        output = {}
        for i, correlation in enumerate(correlations):
            output[str(i)] = float(correlation)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def run_CoalNN_perturb_maf(self):
        print('Starting CoalNN session to perurb MAF..')
        print('Will decode', len(self.random_pairs), 'pairs...')
        self.model.eval()

        num_variants = self.run_dataset.simulations[0]['num_variants']
        phys_pos = -1 * np.ones(num_variants + 2 * self.context_size)
        phys_pos[self.context_size: self.context_size + num_variants] = self.run_dataset.simulations[0]['phys_pos']

        start_pos_range = range(0, num_variants + 2 * self.context_size - self.input_size, self.focus_input_size)

        het_site_average = 0
        het_site_median = 0
        het_site_pos_ratio = 0

        hom_site_average = 0
        hom_site_median = 0
        hom_site_neg_ratio = 0

        counter = 0

        tmrcas_het_sites = []
        tmrcas_hom_sites = []

        with torch.no_grad():
            for iteration, batch in tqdm(enumerate(self.run_dataloader),
                                         total=len(self.run_dataloader),
                                         desc='Batch',
                                         position=0):
                batch_size = batch['label'].shape[0]

                for pos_iteration, start_pos in tqdm(enumerate(start_pos_range),
                                                     total=len(start_pos_range),
                                                     desc='Position',
                                                     position=1):

                    # make input
                    sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                    start_label = pos_iteration * self.focus_input_size
                    sub_batch['label'] = batch['label'][:, start_label: start_label + self.focus_input_size]
                    sub_batch['breakpoints'] = batch['breakpoints'][:, start_label: start_label + self.focus_input_size]
                    sub_batch['pos_index'] = torch.from_numpy(np.asarray(
                        range(start_label, start_label + self.focus_input_size)))

                    # compute prediction
                    self.preprocess_batch(sub_batch)
                    output = self.forward_model(sub_batch)

                    if self.config.log_tmrca:
                        output['output'] = torch.exp(output['output'])
                    threshold = 0.7
                    output['output_const_' + str(threshold)] = self.get_const_piecewise(copy.deepcopy(output),
                                                                                        threshold)

                    # If you want arrays to construct plots, uncomment below: np.save(self.session_name + '/output_'
                    # + str(pos_iteration), output['output'].cpu()) np.save(self.session_name + '/output_const_' +
                    # str(pos_iteration), output['output_const_' + str(threshold)].cpu())

                    """Look at homozygous sites"""
                    perturbed_sub_batch = copy.deepcopy(sub_batch)
                    for pair in range(batch_size):
                        hom_sites = torch.where(sub_batch['input'][pair, 1,
                                                self.context_size: self.context_size + self.focus_input_size] == 1)[0]
                        # Append this to list for homozygous TMRCAs
                        tmrcas_hom_sites.append(torch.flatten(output['output'][:, hom_sites]))

                        # Now perturb
                        for i, site in enumerate(hom_sites):
                            perturbed_sub_batch['input'][pair, 4, self.context_size + site] = torch.max(
                                perturbed_sub_batch['input'][pair, 4, self.context_size + site] - 0.05,
                                torch.tensor(1 / self.run_dataset.num_haplotypes))
                    perturbed_output = self.forward_model(perturbed_sub_batch)

                    if self.config.log_tmrca:
                        perturbed_output['output'] = torch.exp(perturbed_output['output'])
                    perturbed_output['output_const_' + str(threshold)] = self.get_const_piecewise(
                        copy.deepcopy(perturbed_output), threshold)
                    tmrca_change = perturbed_output['output'][:, :] - output['output'][:, :]
                    neg = (tmrca_change < 0).sum() / torch.numel(tmrca_change)
                    hom_site_average += torch.mean(tmrca_change)
                    hom_site_median += torch.median(tmrca_change)
                    hom_site_neg_ratio += neg

                    # If you want arrays to construct plots, uncomment below:
                    # np.save(self.session_name + '/output_hom' + str(pos_iteration), perturbed_output['output'].cpu())
                    # np.save(self.session_name + '/output_const_hom' + str(pos_iteration), perturbed_output['output_const_' + str(threshold)].cpu())

                    """Now look at heterozygous sites"""
                    perturbed_sub_batch = copy.deepcopy(sub_batch)
                    for pair in range(batch_size):
                        het_sites = torch.where(sub_batch['input'][pair, 0,
                                                self.context_size: self.context_size + self.focus_input_size] == 1)[0]
                        # Append this to list for homozygous TMRCAs
                        tmrcas_het_sites.append(torch.flatten(output['output'][:, het_sites]))

                        # Now perturb
                        for i, site in enumerate(het_sites):
                            perturbed_sub_batch['input'][pair, 4, self.context_size + site] = torch.min(
                                perturbed_sub_batch['input'][pair, 4, self.context_size + site] + 0.05,
                                torch.tensor(1 / 2))
                    perturbed_output = self.forward_model(perturbed_sub_batch)

                    if self.config.log_tmrca:
                        perturbed_output['output'] = torch.exp(perturbed_output['output'])
                    perturbed_output['output_const_' + str(threshold)] = self.get_const_piecewise(
                        copy.deepcopy(perturbed_output), threshold)
                    tmrca_change = perturbed_output['output'][:, :] - output['output'][:, :]
                    pos = (tmrca_change > 0).sum() / torch.numel(tmrca_change)
                    het_site_average += torch.mean(tmrca_change)
                    het_site_median += torch.median(tmrca_change)
                    het_site_pos_ratio += pos

                    # If you want arrays to construct plots, uncomment below: perturbed_output['output_const_' + str(
                    # threshold)] = self.get_const_piecewise(copy.deepcopy(perturbed_output), threshold) np.save(
                    # self.session_name + '/output_het' + str(pos_iteration), perturbed_output['output'].cpu())
                    # np.save(self.session_name + '/output_const_het' + str(pos_iteration), perturbed_output[
                    # 'output_const_' + str(threshold)].cpu())

                    counter += 1

                if iteration == 30:
                    break

            tmrcas_hom_sites = torch.flatten(torch.cat(tmrcas_hom_sites)).cpu().numpy()
            tmrcas_het_sites = torch.flatten(torch.cat(tmrcas_het_sites)).cpu().numpy()
            plot_TMRCA_pred_histogram(tmrcas_hom_sites, tmrcas_het_sites, self.session_name)

            print('average_het_TMRCA_change', (het_site_average / counter).cpu())
            print('median_het_TMRCA_change', (het_site_median / counter).cpu())
            print('pos_ratio_het_sites', (het_site_pos_ratio / counter).cpu())

            print('average_hom_TMRCA_change', (hom_site_average / counter).cpu())
            print('median_hom_TMRCA_change', (hom_site_median / counter).cpu())
            print('neg_ratio_hom_sites', (hom_site_neg_ratio / counter).cpu())

            self.save_TMRCA_change_json('average_het_TMRCA_change', (het_site_average / counter).cpu())
            self.save_TMRCA_change_json('median_het_TMRCA_change', (het_site_median / counter).cpu())
            self.save_TMRCA_change_json('pos_ratio_het_sites', (het_site_pos_ratio / counter).cpu())

            self.save_TMRCA_change_json('average_hom_TMRCA_change', (hom_site_average / counter).cpu())
            self.save_TMRCA_change_json('median_hom_TMRCA_change', (hom_site_median / counter).cpu())
            self.save_TMRCA_change_json('neg_ratio_hom_sites', (hom_site_neg_ratio / counter).cpu())

    def compute_saliency_map_CoalNN(self):
        print('Starting CoalNN session to compute saliency maps..')
        print('Will decode', len(self.random_pairs), 'pairs...')
        self.model.eval()

        num_variants = self.run_dataset.simulations[0]['num_variants']
        phys_pos = -1 * np.ones(num_variants + 2 * self.context_size)
        phys_pos[self.context_size: self.context_size + num_variants] = self.run_dataset.simulations[0]['phys_pos']

        for iteration, batch in tqdm(enumerate(self.run_dataloader), total=len(self.run_dataloader)):

            for pos_iteration, start_pos in enumerate(range(0,
                                                            num_variants + 2 * self.context_size - self.input_size,
                                                            self.focus_input_size)):

                # make input
                sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                start_label = pos_iteration * self.focus_input_size
                sub_batch['label'] = batch['label'][:, start_label: start_label + self.focus_input_size]
                sub_batch['breakpoints'] = batch['breakpoints'][:, start_label: start_label + self.focus_input_size]
                sub_batch['pos_index'] = torch.from_numpy(
                    np.asarray(range(start_label, start_label + self.focus_input_size)))
                batch_size = sub_batch['input'].shape[0]

                # compute prediction
                self.preprocess_batch(sub_batch)
                # requires gradients with respect to inputs
                sub_batch['input'].requires_grad_()
                output = self.forward_model(sub_batch)
                tmrcas_output = output['output']
                tmrcas_output_norm = output['saliency_pred']
                output['output_norm'].retain_grad()

                # Use predicted recombination breakpoints
                output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
                prob_recombination = output['breakpoints'][:, 1, :]

                for sample_batch in range(batch_size):

                    breakpoints_positions = torch.arange(0, self.focus_input_size)
                    breakpoints_positions = breakpoints_positions[
                        prob_recombination[sample_batch, :] >= self.config.const_threshold]

                    ibd_lengths = (breakpoints_positions[1:] - breakpoints_positions[:-1]).detach().numpy()
                    unique_ibd_lengths = np.unique(ibd_lengths)
                    quarter = np.percentile(unique_ibd_lengths, 25, interpolation='lower')
                    median = np.percentile(unique_ibd_lengths, 50, interpolation='lower')
                    three_quarter = np.percentile(unique_ibd_lengths, 75, interpolation='lower')
                    max = np.percentile(unique_ibd_lengths, 100, interpolation='lower')

                    quarter_index = np.where(ibd_lengths == quarter)[0][0]  # taking the first match
                    median_index = np.where(ibd_lengths == median)[0][0]
                    three_quarter_index = np.where(ibd_lengths == three_quarter)[0][0]
                    max_index = np.where(ibd_lengths == max)[0][0]

                    ibd_start_index_positions = [quarter_index, median_index, three_quarter_index, max_index]

                    for ibd_index_start in ibd_start_index_positions:
                        ibd_start = breakpoints_positions[ibd_index_start]
                        ibd_end = breakpoints_positions[ibd_index_start + 1]

                        # get a saliency map per IBD segment
                        grad_index = torch.zeros_like(tmrcas_output)
                        # sum gradients across the IBD segment
                        grad_index[sample_batch, ibd_start: ibd_end] = 1

                        # compute gradients
                        tmrcas_output.backward(grad_index, retain_graph=True)
                        tmrcas_output_norm.backward(grad_index, retain_graph=True)
                        saliency = sub_batch['input'].grad.data[sample_batch, :, :]
                        saliency_norm = output['output_norm'].grad.data[sample_batch, :, :]

                        # infer age of IBD segment
                        median_ibd_age = int(torch.exp(tmrcas_output[sample_batch, ibd_start:ibd_end]).median())

                        if median_ibd_age < self.config.tmrca_lower_bound:
                            name_start = int(ibd_start + self.context_size)
                            name_end = int(ibd_end + self.context_size)
                            save_torch(self.session_name,
                                       saliency,
                                       'saliency_' + str(sample_batch) + '_age_' + str(median_ibd_age) +
                                       '_' + str(name_start) + '_to_' + str(name_end))
                            save_torch(self.session_name,
                                       sub_batch['input'][sample_batch, :, :],
                                       'input_' + str(sample_batch) + '_age_' + str(median_ibd_age) +
                                       '_' + str(name_start) + '_to_' + str(name_end))
                            save_torch(self.session_name,
                                       saliency_norm,
                                       'saliency_norm_' + str(sample_batch) + '_age_' + str(median_ibd_age) +
                                       '_' + str(name_start) + '_to_' + str(name_end))
                            save_torch(self.session_name,
                                       output['output_norm'][sample_batch, :, :],
                                       'input_norm_' + str(sample_batch) + '_age_' + str(median_ibd_age) +
                                       '_' + str(name_start) + '_to_' + str(name_end))
                            save_numpy(self.session_name,
                                       phys_pos[start_pos: start_pos + self.input_size],
                                       'phys_pos_' + str(sample_batch) + '_age_' + str(median_ibd_age) +
                                       '_' + str(name_start) + '_to_' + str(name_end))

                        sub_batch['input'].grad.data.zero_()
                        output['output_norm'].grad.data.zero_()
                        # self.visualize_saliency()

    def compute_saliency_map_per_site(self):
        print('Starting CoalNN session to compute saliency maps per site..')
        print('Will decode', len(self.random_pairs), 'pairs...')
        self.model.eval()

        num_variants = self.run_dataset.simulations[0]['num_variants']
        phys_pos = -1 * np.ones(num_variants + 2 * self.context_size)
        phys_pos[self.context_size: self.context_size + num_variants] = self.run_dataset.simulations[0]['phys_pos']

        for iteration, batch in tqdm(enumerate(self.run_dataloader), total=len(self.run_dataloader)):

            for pos_iteration, start_pos in enumerate(range(0,
                                                            num_variants + 2 * self.context_size - self.input_size,
                                                            self.focus_input_size)):

                # make input
                sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                start_label = pos_iteration * self.focus_input_size
                sub_batch['label'] = batch['label'][:, start_label: start_label + self.focus_input_size]
                sub_batch['breakpoints'] = batch['breakpoints'][:, start_label: start_label + self.focus_input_size]
                sub_batch['pos_index'] = torch.from_numpy(
                    np.asarray(range(start_label, start_label + self.focus_input_size)))
                batch_size = sub_batch['input'].shape[0]

                # compute prediction
                self.preprocess_batch(sub_batch)
                # requires gradients with respect to inputs
                sub_batch['input'].requires_grad_()
                output = self.forward_model(sub_batch)
                # Multiply by 1e6 to avoid zeros in gradient
                tmrcas_output = (1e6) * output['output']
                tmrcas_output_norm = (1e6) * output['saliency_pred']
                output['output_norm'].retain_grad()

                for sample_batch in range(batch_size):
                    # get a saliency map for the most recent predicted TMRCA
                    most_recent_site = torch.argmax(tmrcas_output[sample_batch, :])
                    grad_index = torch.zeros_like(tmrcas_output)
                    # sum gradients across the IBD segment
                    grad_index[sample_batch, most_recent_site] = 1

                    # compute gradients
                    tmrcas_output.backward(grad_index, retain_graph=True)
                    tmrcas_output_norm.backward(grad_index, retain_graph=True)
                    saliency = sub_batch['input'].grad.data[sample_batch, :, :]
                    saliency_norm = output['output_norm'].grad.data[sample_batch, :, :]

                    # get age of IBD segment
                    site_age = int(torch.exp(output['output'][sample_batch, most_recent_site]))
                    if site_age > self.config.tmrca_upper_bound:
                        position = phys_pos[most_recent_site]
                        name_start = int(position)
                        name_end = int(position)
                        save_torch(self.session_name,
                                   saliency,
                                   'saliency_' + str(sample_batch) + '_age_' + str(site_age) +
                                   '_' + str(name_start) + '_to_' + str(name_end))
                        save_torch(self.session_name,
                                   sub_batch['input'][sample_batch, :, :],
                                   'input_' + str(sample_batch) + '_age_' + str(site_age) +
                                   '_' + str(name_start) + '_to_' + str(name_end))
                        save_torch(self.session_name,
                                   saliency_norm,
                                   'saliency_norm_' + str(sample_batch) + '_age_' + str(site_age) +
                                   '_' + str(name_start) + '_to_' + str(name_end))
                        save_torch(self.session_name,
                                   output['output_norm'][sample_batch, :, :],
                                   'input_norm_' + str(sample_batch) + '_age_' + str(site_age) +
                                   '_' + str(name_start) + '_to_' + str(name_end))
                        save_numpy(self.session_name,
                                   phys_pos[start_pos: start_pos + self.input_size],
                                   'phys_pos_' + str(sample_batch) + '_age_' + str(site_age) +
                                   '_' + str(name_start) + '_to_' + str(name_end))

                    sub_batch['input'].grad.data.zero_()
                    output['output_norm'].grad.data.zero_()
                    # self.visualize_saliency()

    @track
    def run_CoalNN(self):
        print('Starting CoalNN session..')
        print('Will decode', len(self.random_pairs), 'pairs...')
        start_time = time()
        self.model.eval()

        self.tot_elts = 0

        with torch.no_grad():
            if self.config.eval:
                num_thresholds = len(self.const_thresholds) + 1  # + 1 for threshold = 0
                run_score = num_thresholds * [0]  # first entry for raw output
                run_loss = num_thresholds * [0]  # first entry for raw output
                run_segments = num_thresholds * [0]  # first entry for raw output
                run_ground_truth_segments = 0
                # run_nb_is_valid = 0

            num_variants = self.run_dataset.simulations[0]['num_variants']

            for iteration, batch in tqdm(enumerate(self.run_dataloader), total=len(self.run_dataloader)):
                for pos_iteration, start_pos in enumerate(range(0,
                                                                num_variants + 2 * self.context_size - self.input_size,
                                                                self.focus_input_size)):
                    sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}

                    if self.config.eval:
                        start_label = pos_iteration * self.focus_input_size
                        sub_batch['label'] = batch['label'][:, start_label: start_label + self.focus_input_size]
                        sub_batch['breakpoints'] = batch['breakpoints'][:,
                                                   start_label: start_label + self.focus_input_size]
                        sub_batch['pos_index'] = torch.from_numpy(
                            np.asarray(range(start_label, start_label + self.focus_input_size)))
                        # score, loss, segments, nb_is_valid = self.run_step(sub_batch)
                        score, loss, segments, ground_truth_segments = self.run_step(sub_batch)
                        run_score = [sum(x) for x in zip(score, run_score)]
                        run_loss = [sum(x) for x in zip(loss, run_loss)]
                        run_segments = [sum(x) for x in zip(segments, run_segments)]
                        run_ground_truth_segments += ground_truth_segments
                        # run_nb_is_valid += nb_is_valid
                    else:
                        self.run_prediction(sub_batch)

                # run last bin if necessary
                if self.config.run_last_bin:
                    if num_variants % self.focus_input_size != 0:
                        start_pos = num_variants + 2 * self.context_size - self.input_size
                        sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}

                        if self.config.eval:
                            last_position_processed = start_label + self.focus_input_size
                            start_label = num_variants - self.focus_input_size
                            # print("last bin start_pos", start_pos, "start_label", start_label)
                            sub_batch['label'] = batch['label'][:, start_label: num_variants]
                            sub_batch['breakpoints'] = batch['breakpoints'][:, start_label: num_variants]
                            sub_batch['pos_index'] = torch.from_numpy(np.asarray(range(start_label, num_variants)))
                            last_position_processed_idx = last_position_processed - start_label
                            # score, loss, segments, nb_is_valid = self.run_step(sub_batch, start_pos=last_position_processed_idx)
                            score, loss, segments, ground_truth_segments = self.run_step(sub_batch,
                                                                                         start_pos=last_position_processed_idx)
                            run_score = [sum(x) for x in zip(score, run_score)]
                            run_loss = [sum(x) for x in zip(loss, run_loss)]
                            run_segments = [sum(x) for x in zip(segments, run_segments)]
                            run_ground_truth_segments += ground_truth_segments
                            # run_nb_is_valid += nb_is_valid
                        else:
                            self.run_prediction(sub_batch)

            if self.config.eval:
                run_score = [score / (len(self.random_pairs) * num_variants) for score in run_score]
                run_loss = [loss / (len(self.random_pairs) * num_variants) for loss in run_loss]
                # run_score = [score / run_nb_is_valid for score in run_score]
                # run_loss = [loss / run_nb_is_valid for loss in run_loss]
                print('Total number of segments in ground truth:', run_ground_truth_segments, '\n')
                self.save_segments_json('ground_truth_nb_segments', run_ground_truth_segments)
                print('L1 loss CoalNN: {:.3f}'.format(run_score[0]))
                print('L2 loss CoalNN: {:.3f}'.format(sqrt(run_loss[0])))
                self.save_metrics_json('CoalNN', run_score[0], sqrt(run_loss[0]))
                print('Total number of segments in prediction:', run_segments[0], '\n')
                self.save_segments_json('nb_segments', run_segments[0])
                if self.config.constant_piecewise_output:
                    for i, threshold in enumerate(self.const_thresholds, 1):
                        print('L1 loss CoalNN constant piecewise {}: {:.3f}'.format(threshold, run_score[i]))
                        print('L2 loss CoalNN constant piecewise {}: {:.3f}'.format(threshold, sqrt(run_loss[i])))
                        self.save_metrics_json('CoalNN_const_' + str(threshold), run_score[i], sqrt(run_loss[i]))
                        print('Total number of segments in predicted constant piecewise {}: {}\n'.format(threshold,
                                                                                                         run_segments[
                                                                                                             i]))
                        self.save_segments_json('nb_segments_const_' + str(threshold), run_segments[i])
                if self.config.scatter_plot:
                    save_numpy(self.session_name, self.TMRCA_LABEL, 'coalNN_tmrca_label')
                    save_numpy(self.session_name, self.TMRCA_PREDICTION, 'coalNN_tmrca_prediction')
                    save_numpy(self.session_name, self.MAF, 'coalNN_maf')
                    if self.config.constant_piecewise_output:
                        for i, threshold in enumerate(self.const_thresholds, 1):
                            save_numpy(self.session_name, self.TMRCA_PREDICTION_CONST, 'coalNN_const_tmrca_prediction')
                # print('Number is_valid', run_nb_is_valid)

            time_duration = time() - start_time
            print('CoalNN decoding done in : {:.0f}ms'.format(1000 * time_duration))

    def run_prediction(self, batch):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        return output

    @staticmethod
    def get_number_segments(label, output, threshold):
        # ground_truth = torch.sum(label['breakpoints']).item()  # return number of recombination points (i.e segments)
        prediction_breakpoints = output['breakpoints'][:, 1, :]  # class 1
        # prediction_breakpoints = output['breakpoints']
        prediction = torch.sum((prediction_breakpoints >= threshold).type(torch.uint8)).item()
        return prediction

    def run_step(self, batch, start_pos=None):

        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
        if self.config.log_tmrca:
            output['output'] = torch.exp(output['output'])
            batch['label'] = torch.exp(batch['label'])
            # output['output'] = torch.pow(10, output['output'])
            # batch['label'] = torch.pow(10, batch['label'])
        if self.config.constant_piecewise_output:
            for threshold in self.const_thresholds:
                output['output_const_' + str(threshold)] = self.get_const_piecewise(copy.deepcopy(output), threshold)

        if start_pos is not None:
            batch['label'] = batch['label'][:, start_pos:]
            batch['breakpoints'] = batch['breakpoints'][:, start_pos:]
            output['output'] = output['output'][:, start_pos:]
            output['breakpoints'] = output['breakpoints'][:, :, start_pos:]
            if self.config.constant_piecewise_output:
                for threshold in self.const_thresholds:
                    output['output_const_' + str(threshold)] = output['output_const_' + str(threshold)][:, start_pos:]

        self.tot_elts += torch.numel(batch['label'])

        # is_valid = batch['label'] < 10000000000
        # batch['label'] = batch['label'][is_valid]
        # output['output'] = output['output'][is_valid]
        # nb_is_valid = torch.sum(is_valid).item()

        num_thresholds = len(self.const_thresholds) + 1  # + 1 for threshold = 0
        score, loss, segments = num_thresholds * [0], num_thresholds * [0], num_thresholds * [0]
        score[0] = self.forward_metric(batch['label'], output['output']).item()
        loss[0] = self.forward_loss(batch['label'], output['output']).item()
        segments[0] = self.get_number_segments(batch, output, threshold=0)
        ground_truth_segments = torch.sum(batch['breakpoints']).item()

        if self.config.scatter_plot:
            self.TMRCA_LABEL.append(batch['label'].cpu().numpy())
            self.TMRCA_PREDICTION.append(output['output'].cpu().numpy())
            self.MAF.append(batch['input'][:, 4, self.context_size:-self.context_size].cpu().numpy())
            if self.config.constant_piecewise_output:
                for threshold in self.const_thresholds:
                    self.TMRCA_PREDICTION_CONST.append(output['output_const_' + str(threshold)].cpu().numpy())

        if self.config.constant_piecewise_output:
            for i, threshold in enumerate(self.const_thresholds, 1):
                score[i] = self.forward_metric(batch['label'], output['output_const_' + str(threshold)]).item()
                loss[i] = self.forward_loss(batch['label'], output['output_const_' + str(threshold)]).item()
                segments[i] = self.get_number_segments(batch, output, threshold=threshold)

        return score, loss, segments, ground_truth_segments

    def save_metrics_json(self, name, l1, l2):
        filename = os.path.join(self.session_name, name + '_metrics.json')
        output = {'l1': float("{:.3f}".format(l1)), 'l2': float("{:.3f}".format(l2))}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def save_segments_json(self, name, prediction):
        filename = os.path.join(self.session_name, name + '_metrics.json')
        output = {'nb_segments': prediction}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

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

        if self.options.simulation_size:
            config_run['sample_size'] = int(self.options.size)

        if self.options.ref_size:
            config_run['val_ref_size'] = int(self.options.ref_size)

        if self.options.const_threshold:
            config_run['constant_piecewise_output'] = True
            config_run['const_threshold'] = float(self.options.const_threshold)

        if self.options.downsample_size:
            config_run['downsample_size'] = int(self.options.downsample_size)

        if self.options.batch_size:
            config_run['batch_size'] = self.options.batch_size

        # if ('visualise_first_layer' in config_run):
        #     if config_run['visualise_first_layer']:
        #         config_run['data_type'] = 'impute'

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

        populations = ["CEU.Terhorst", "ACB", "CEU", "ESN", "GWD", "KHV", "PEL", "STU", "ASW", "CHB", "FIN", "IBS",
                       "LWK", "PJL", "TSI", "BEB", "CHS", "GBR", "ITU", "MSL", "PUR", "YRI", "CDX",
                       "CLM", "GIH", "JPT", "MXL"]
        if "constant" in self.config.demography:
            demo = 'constant' + '.Ne.' + str(self.config.Ne)
        elif self.config.demography in populations:
            demo = str(self.config.demography)
        else:
            raise ValueError('Demographic model is unknown.')

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
                       + '.' + str(self.config.chr_length) + 'Mbp' \
                       + '.S_run.' + str(self.config.sample_size_run) \
                       + '.sample_size.' + str(self.config.sample_size) \
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

        with open(os.path.join(self.session_name, 'config_run.yml'), 'w') as f:
            yaml.dump(config_run, f)

        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f)

        random.seed(self.config.seed_run)
        np.random.seed(self.config.seed_run)

    def save_TMRCA_change_json(self, name, val):
        filename = os.path.join(self.session_name, name + '_metrics.json')
        output = {'value': float("{:.3f}".format(val))}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
