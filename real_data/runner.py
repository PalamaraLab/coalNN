"""
Code adapted from Anthony Hu.
https://github.com/anthonyhu/ml-research
"""

import os
import sys
import yaml
import torch
import copy
import pickle
import numpy as np
import torch.nn.functional as F
import datetime
import socket
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
from real_data.data import RealDataset
from models.fc_network import FcNetwork
from models.cnn_network import CnnNetwork
from models.transformer_network import TransformerNetwork
from utils import Config, Logger, print_model_spec, track
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import gmean
from utils import save_numpy


class Runner:
    __meta_class__ = ABCMeta

    def __init__(self, options):
        self.options = options

        ##########
        # Restore session
        ##########
        self.config = None
        self.config_run = None
        self.restore_session_name = ''

        ##########
        # Running sessions
        ##########
        self.session_name = ''
        self.initialise_session()

        ##########
        # Model
        ##########
        self.model = None
        self.global_step = 1
        self.device = None

        ##########
        # Metrics
        ##########
        self.run_metrics = None
        self.create_metrics()

        ##########
        # Data
        ##########
        self.random_pairs = None
        self.const_thresholds = None
        self.run_dataset = None
        self.run_dataloader = None
        self.input_size, self.focus_input_size, self.context_size, self.x_dim = None, None, None, None
        self.root_time = None
        self.populations = ["ACB", "CEU", "ESN", "GWD", "KHV", "PEL", "STU", "ASW", "CHB",
                            "FIN", "IBS", "LWK", "PJL", "TSI", "BEB", "CHS", "GBR", "ITU",
                            "MSL", "PUR", "YRI", "CDX", "CLM", "GIH", "JPT", "MXL"]

        if hasattr(self.config, 'const_threshold'):
            self.const_threshold = self.config.const_threshold

    @abstractmethod
    def create_data(self, population):
        """Create run datasets and dataloaders."""
        print('-' * 100)
        print('Building dataset...')

        self._get_attributes()

        self.run_dataset = RealDataset(self.config,
                                       demo=population,
                                       input_size=self.input_size,
                                       focus_input_size=self.focus_input_size,
                                       context_size=self.context_size)
        print('Dataset has {} pairs of haplotypes.\n'.format(len(self.run_dataset)))

        self.run_dataloader = DataLoader(self.run_dataset,
                                         self.config.batch_size,
                                         num_workers=self.config.n_workers)

    def _get_attributes(self):
        self.x_dim = self.config.x_dim
        self.x_dim[0] = self.config.batch_size
        self.input_size = self.x_dim[-1]
        self.focus_input_size = self.config.focus_input_size
        self.context_size = self.config.context_size
        self.root_time = self.config.root_time

    @abstractmethod
    def create_model(self):
        """Build the neural network."""
        print('-' * 100)
        print('Building model {}...'.format(self.config.model.network))
        print('Batches have shape {}.'.format(self.x_dim))
        if self.config.model.network == 'fc':
            self.model = FcNetwork(self.x_dim, self.config)
        elif self.config.model.network == 'cnn':
            if self.config.model.restriction_activation:
                clamp = False
            else:
                clamp = True
            self.model = CnnNetwork(self.focus_input_size, self.x_dim, self.config, self.root_time, clamp)
        elif self.config.model.network == 'transformer':
            self.model = TransformerNetwork(self.x_dim, self.config)

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
        return self.model(batch)

    @abstractmethod
    def forward_loss(self, batch, output):
        """Compute the loss."""

    @abstractmethod
    def forward_metric(self, batch, output):
        """Compute the metric."""

    def run(self):
        print('Running session..')
        restore_path = self.config_run['restore_path']
        population = self.config_run['demo']
        # Restore session
        self.restore_session(restore_path)
        # Create data
        self.create_data(population=population)
        # Create model
        self.create_model()
        print_model_spec(self.model)
        self.model.to(self.device)
        # Restore model
        self.load_checkpoint(restore_path)
        # Estimate allele ages
        concordant_ages, discordant_ages, mean_ages, \
        n_concordant_pairs, n_discordant_pairs, \
        n_concordant_pairs_post_filtering, n_discordant_pairs_post_filtering = \
            self.date_variants_deep_coalescent()
        # Compute quality score
        quality_score = self.compute_quality_score(n_concordant_pairs, n_discordant_pairs,
                                                   n_concordant_pairs_post_filtering, n_discordant_pairs_post_filtering)
        # Store features
        phys_dist = self.run_dataset.phys_dist.reshape(-1)
        gen_dist = self.run_dataset.gen_dist.reshape(-1)
        maf = self.run_dataset.maf.reshape(-1)
        # Add NaN values for non-polymorphic variants
        is_concordant_nan = np.isnan(concordant_ages)
        is_discordant_nan = np.isnan(discordant_ages)
        is_mean_nan = np.isnan(mean_ages)
        concordant_ages[is_concordant_nan] = -1
        discordant_ages[is_discordant_nan] = -1
        mean_ages[is_mean_nan] = -1
        concordant_ages = self.add_non_poly_variants(concordant_ages)
        discordant_ages = self.add_non_poly_variants(discordant_ages)
        mean_ages = self.add_non_poly_variants(mean_ages)
        quality_score = self.add_non_poly_variants(quality_score)
        phys_dist = self.add_non_poly_variants(phys_dist)
        gen_dist = self.add_non_poly_variants(gen_dist)
        maf = self.add_non_poly_variants(maf)
        n_concordant_pairs = self.add_non_poly_variants(n_concordant_pairs)
        n_discordant_pairs = self.add_non_poly_variants(n_discordant_pairs)
        n_concordant_pairs_post_filtering = \
            self.add_non_poly_variants(n_concordant_pairs_post_filtering)
        n_discordant_pairs_post_filtering = \
            self.add_non_poly_variants(n_discordant_pairs_post_filtering)
        # Save output file
        self.save_annotations_file(population, maf, concordant_ages,
                                   discordant_ages, mean_ages,
                                   n_concordant_pairs, n_discordant_pairs,
                                   n_concordant_pairs_post_filtering,
                                   n_discordant_pairs_post_filtering,
                                   quality_score)

        # save_numpy(self.session_name, concordant_ages, 'concordant_ages')
        # save_numpy(self.session_name, discordant_ages, 'discordant_ages')
        # save_numpy(self.session_name, mean_ages, 'mean_ages')
        # save_numpy(self.session_name, phys_dist, 'phys_dist')
        # save_numpy(self.session_name, gen_dist, 'gen_dist')
        # save_numpy(self.session_name, maf, 'maf')
        # save_numpy(self.session_name, self.run_dataset.all_phys_pos, 'phys_pos')
        # save_numpy(self.session_name, self.run_dataset.ref_alleles, 'ref_alleles')
        # save_numpy(self.session_name, self.run_dataset.alt_alleles, 'alt_alleles')
        # save_numpy(self.session_name, self.run_dataset.anc_alleles, 'anc_alleles')
        # save_numpy(self.session_name, self.run_dataset.rsids, 'rsids')

    @staticmethod
    def compute_quality_score(n_concordant_pairs, n_discordant_pairs,
                              n_concordant_pairs_post_filtering, n_discordant_pairs_post_filtering):
        n_before_filtering = n_concordant_pairs + n_discordant_pairs
        n_after_filtering = n_concordant_pairs_post_filtering + n_discordant_pairs_post_filtering
        return n_after_filtering / n_before_filtering

    def save_annotations_file(self, population, maf, concordant_ages, discordant_ages, mean_ages,
                              n_concordant_pairs, n_discordant_pairs, n_concordant_pairs_post_filtering,
                              n_discordant_pairs_post_filtering, quality_score):
        file_name = os.path.join(self.session_name, 'chr' + str(self.config.chr) + '.'
                                 + population + '.coalNN_age_annotations.txt')
        print("Saving annotations in " + file_name + "...")
        with open(file_name, 'w') as file_annotations:
            file_annotations.write('chromosome\trsid\tstart(bp)\tend(bp)\treference_state\talternate_state' +
                                   '\tancestral_state\t' + population + '_AC\t' + population + '_AF\t' + population +
                                   '_AC_derived\t' + population + '_AF_derived\tmaf\tlower_age\tupper_age\t' +
                                   'age_estimate\tconcordant_pairs\tdiscordant_pairs')
            if self.config.filter_outliers:
                file_annotations.write('\tconcordant_pairs_filtering\tdiscordant_pairs_filtering')
                file_annotations.write('\tQS')
            file_annotations.write('\n')
            for i, variant in enumerate(self.run_dataset.all_phys_pos):
                if self.run_dataset.is_polymorphic_variants[i]:
                    file_annotations.write(str(self.config.chr) + '\t')
                    file_annotations.write(str(self.run_dataset.rsids[i]) + '\t')
                    file_annotations.write(str(int(variant)) + '\t')
                    file_annotations.write(str(int(variant + 1)) + '\t')
                    file_annotations.write(str(self.run_dataset.ref_alleles[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.alt_alleles[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.anc_alleles[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.AC[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.AF[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.AC_derived[i]) + '\t')
                    file_annotations.write(str(self.run_dataset.AF_derived[i]) + '\t')
                    file_annotations.write(str(maf[i]) + '\t')
                    file_annotations.write(str(concordant_ages[i]) + '\t')
                    file_annotations.write(str(discordant_ages[i]) + '\t')
                    file_annotations.write(str(mean_ages[i]) + '\t')
                    n_con_pairs = n_concordant_pairs[i]
                    n_disc_pairs = n_discordant_pairs[i]
                    if n_con_pairs == n_con_pairs:
                        # is not nan, we can cast to int
                        n_con_pairs = int(n_con_pairs)
                    if n_disc_pairs == n_disc_pairs:
                        # is not nan, we can cast to int
                        n_disc_pairs = int(n_disc_pairs)
                    file_annotations.write(str(n_con_pairs) + '\t')
                    file_annotations.write(str(n_disc_pairs))
                    if self.config.filter_outliers:
                        n_con_pairs = n_concordant_pairs_post_filtering[i]
                        n_disc_pairs = n_discordant_pairs_post_filtering[i]
                        if n_con_pairs == n_con_pairs:
                            # is not nan, we can cast to int
                            n_con_pairs = int(n_con_pairs)
                        if n_disc_pairs == n_disc_pairs:
                            # is not nan, we can cast to int
                            n_disc_pairs = int(n_disc_pairs)
                        file_annotations.write('\t' + str(n_con_pairs) + '\t')
                        file_annotations.write(str(n_disc_pairs) + '\t')
                        file_annotations.write(str(quality_score[i]))
                    file_annotations.write('\n')
        return

    def add_non_poly_variants(self, array):
        # add nan values to non polymorphic variants
        all_pos_array = float('nan') * np.ones(self.run_dataset.num_all_variants)
        i_polymorphic_variant = 0
        for i, variant in enumerate(self.run_dataset.all_phys_pos):
            if self.run_dataset.is_polymorphic_variants[i]:
                all_pos_array[i] = array[i_polymorphic_variant]
                i_polymorphic_variant += 1
        return all_pos_array

    def save_ages_pickle(self, name, ages):
        filename = os.path.join(self.session_name, name + '_allele_ages.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(ages, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @track
    def date_variants_deep_coalescent(self):
        print('Starting CoalNN session..')
        print('Will decode', self.run_dataset.num_pairs, 'pairs...')
        start_time = time()
        self.model.eval()

        with torch.no_grad():

            num_variants = self.run_dataset.num_variants
            concordant_ages = float('nan') * np.ones(num_variants)
            discordant_ages = float('nan') * np.ones(num_variants)
            n_concordant_pairs = float('nan') * np.ones(num_variants)
            n_discordant_pairs = float('nan') * np.ones(num_variants)
            n_concordant_pairs_post_filtering = float('nan') * np.ones(num_variants)
            n_discordant_pairs_post_filtering = float('nan') * np.ones(num_variants)
            start_pos_range = range(0, num_variants - self.focus_input_size, self.focus_input_size)
            singletons = np.sum(self.run_dataset.genotype_matrix, axis=0) == 1

            for pos_iteration, start_pos in tqdm(enumerate(start_pos_range),
                                                 total=len(start_pos_range),
                                                 desc='Position',
                                                 position=0):

                # if pos_iteration >= 1:
                #     break

                self.run_dataset.update_positions(start_pos)
                # move to CPU because of RAM issues with GWD and YRI populations
                concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size),
                                                  device=self.device)
                discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size),
                                                  device=self.device)
                # concordant_ages_bin = np.zeros((self.run_dataset.num_pairs, self.focus_input_size))
                # discordant_ages_bin = np.zeros((self.run_dataset.num_pairs, self.focus_input_size))
                prev_batch_size = self.config.batch_size
                for iteration, batch in tqdm(enumerate(self.run_dataloader),
                                             total=len(self.run_dataloader),
                                             desc='Batch',
                                             leave=False,
                                             position=1):
                    # sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                    prediction = self.run_prediction(batch)
                    concordant_prediction, discordant_prediction = \
                        self.compute_concordant_discordant_prediction(batch, prediction)
                    # pdb.set_trace()
                    cur_batch_size = prediction.shape[0]
                    concordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] = \
                        concordant_prediction
                    discordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] = \
                        discordant_prediction
                    prev_batch_size = cur_batch_size

                concordant_ages_bin = concordant_ages_bin.cpu().numpy()
                discordant_ages_bin = discordant_ages_bin.cpu().numpy()

                # set concordant ages to NaN for singletons
                # set 1 concordant age to '0' for singletons, (just one of them to avoid bias)
                singletons_bin = singletons[start_pos: start_pos + self.focus_input_size]
                concordant_ages_bin[:, singletons_bin] = float('nan')
                concordant_ages_bin[0, singletons_bin] = 0

                # compute number of concordant / discordant pairs
                n_concordant_pairs[start_pos: start_pos + self.focus_input_size] = \
                    np.sum(~np.isnan(concordant_ages_bin), axis=0)
                n_concordant_pairs[start_pos: start_pos + self.focus_input_size][singletons_bin] = 0
                n_discordant_pairs[start_pos: start_pos + self.focus_input_size] = \
                    np.sum(~np.isnan(discordant_ages_bin), axis=0)

                if self.config.filter_outliers:
                    concordant_ages_bin, discordant_ages_bin = self.filter_outliers(concordant_ages_bin,
                                                                                    discordant_ages_bin,
                                                                                    self.focus_input_size)
                    n_concordant_pairs_post_filtering[start_pos: start_pos + self.focus_input_size] = \
                        np.sum(~np.isnan(concordant_ages_bin), axis=0)
                    n_concordant_pairs_post_filtering[start_pos: start_pos + self.focus_input_size][singletons_bin] = 0
                    n_discordant_pairs_post_filtering[start_pos: start_pos + self.focus_input_size] = \
                        np.sum(~np.isnan(discordant_ages_bin), axis=0)

                concordant_ages[start_pos: start_pos + self.focus_input_size] = np.nanmax(concordant_ages_bin, axis=0)
                discordant_ages[start_pos: start_pos + self.focus_input_size] = np.nanmin(discordant_ages_bin, axis=0)

            # run last bin
            print('Decoding last bin...')
            start_pos = num_variants - self.focus_input_size
            self.run_dataset.update_positions(start_pos)
            # move to CPU because of RAM issues with GWD and YRI populations
            concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size),
                                              device=self.device)
            discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size),
                                              device=self.device)
            # concordant_ages_bin = np.zeros((self.run_dataset.num_pairs, self.focus_input_size))
            # discordant_ages_bin = np.zeros((self.run_dataset.num_pairs, self.focus_input_size))
            prev_batch_size = self.config.batch_size
            for iteration, batch in tqdm(enumerate(self.run_dataloader),
                                         total=len(self.run_dataloader),
                                         desc='Batch',
                                         leave=False,
                                         position=1):
                # sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                prediction = self.run_prediction(batch)
                concordant_prediction, discordant_prediction = \
                    self.compute_concordant_discordant_prediction(batch, prediction)
                cur_batch_size = prediction.shape[0]
                concordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] \
                    = concordant_prediction
                discordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] \
                    = discordant_prediction
                prev_batch_size = cur_batch_size
            concordant_ages_bin = concordant_ages_bin.cpu().numpy()
            discordant_ages_bin = discordant_ages_bin.cpu().numpy()

            # set concordant ages to NaN for singletons
            # set 1 concordant age to '0' for singletons, (just one of them to avoid bias)
            start_label = num_variants - self.focus_input_size
            singletons_bin = singletons[start_label: start_label + self.focus_input_size]
            concordant_ages_bin[:, singletons_bin] = float('nan')
            concordant_ages_bin[0, singletons_bin] = 0

            # compute number of concordant / discordant pairs
            n_concordant_pairs[start_label: start_label + self.focus_input_size] = \
                np.sum(~np.isnan(concordant_ages_bin), axis=0)
            n_concordant_pairs[start_label: start_label + self.focus_input_size][singletons_bin] = 0
            n_discordant_pairs[start_label: start_label + self.focus_input_size] = \
                np.sum(~np.isnan(discordant_ages_bin), axis=0)

            if self.config.filter_outliers:
                concordant_ages_bin, discordant_ages_bin = self.filter_outliers(concordant_ages_bin,
                                                                                discordant_ages_bin,
                                                                                self.focus_input_size)
                n_concordant_pairs_post_filtering[start_label: start_label + self.focus_input_size] = \
                    np.sum(~np.isnan(concordant_ages_bin), axis=0)
                n_concordant_pairs_post_filtering[start_label: start_label + self.focus_input_size][singletons_bin] = 0
                n_discordant_pairs_post_filtering[start_label: start_label + self.focus_input_size] = \
                    np.sum(~np.isnan(discordant_ages_bin), axis=0)

            concordant_ages[start_label: start_label + self.focus_input_size] = np.nanmax(concordant_ages_bin, axis=0)
            discordant_ages[start_label: start_label + self.focus_input_size] = np.nanmin(discordant_ages_bin, axis=0)

            if self.config.dating_mode == 'arithmetic':
                variants_ages = np.mean([concordant_ages, discordant_ages], axis=0)
            elif self.config.dating_mode == 'geometric':
                variants_ages = gmean([concordant_ages, discordant_ages], axis=0)
            else:
                raise ValueError('Dating mode for CoalNN is unknown.')

            # take care of edge cases
            is_discordant_ages_nan = np.isnan(discordant_ages)
            is_concordant_ages_nan = np.isnan(concordant_ages)
            # if discordant age is Nan, but concordant age is not NaN and it is not a singleton, mean is concordant age
            criterion = is_discordant_ages_nan & ~is_concordant_ages_nan & ~singletons
            variants_ages[criterion] = concordant_ages[criterion]
            # if concordant age is NaN and it is not a singleton, but discordant age is not NaN, mean is discordant age
            criterion = is_concordant_ages_nan & ~singletons & ~is_discordant_ages_nan
            variants_ages[criterion] = discordant_ages[criterion]

            time_duration = time() - start_time
            print('CoalNN decoding done in : {:.0f}ms'.format(1000 * time_duration))
            return concordant_ages, discordant_ages, variants_ages, n_concordant_pairs, n_discordant_pairs, \
                   n_concordant_pairs_post_filtering, n_discordant_pairs_post_filtering

    def filter_outliers(self, concordant_ages, discordant_ages, bin_size):
        # print("Filtering outliers...")
        clf = AdaBoostClassifier(n_estimators=1, random_state=0)
        # clf = SVC(kernel='linear')
        for variant in tqdm(range(bin_size),
                            desc='Filtering outliers',
                            leave=False,
                            position=2):
            self.filter_outliers_axis(concordant_ages[:, variant], discordant_ages[:, variant], clf)
        return concordant_ages, discordant_ages

    @staticmethod
    def filter_outliers_axis(concordant_ages, discordant_ages, clf):
        concordant_ages_non_nan_index = np.logical_not(np.isnan(concordant_ages))
        concordant_ages_non_nan = concordant_ages[concordant_ages_non_nan_index]
        discordant_ages_non_nan_index = np.logical_not(np.isnan(discordant_ages))
        discordant_ages_non_nan = discordant_ages[discordant_ages_non_nan_index]
        if len(discordant_ages_non_nan) > 0 and len(concordant_ages_non_nan) > 0:
            # non-singleton
            X = np.concatenate((discordant_ages_non_nan, concordant_ages_non_nan)).reshape(-1, 1)
            Y = np.concatenate((np.ones_like(discordant_ages_non_nan), np.zeros_like(concordant_ages_non_nan)))
            clf.fit(X, Y)
            decision_tree = clf.estimators_[0].tree_
            thresholds = decision_tree.threshold[decision_tree.feature == 0]
            if len(thresholds) > 0:
                best_threshold = thresholds[0]
                concordant_ages[concordant_ages > best_threshold] = float('nan')
                discordant_ages[discordant_ages < best_threshold] = float('nan')

    def compute_concordant_discordant_prediction(self, batch, prediction):
        # Issue of prediction <1 improved by adding telomeres + centromeres of chromosomes to the simulations
        prediction[prediction < 1] = float('nan')
        concordant_pairs = torch.tensor(batch['input'][:, 1, self.context_size:-self.context_size],
                                        dtype=bool,
                                        device=self.device)  # AND gate
        discordant_pairs = torch.tensor(batch['input'][:, 0, self.context_size:-self.context_size],
                                        dtype=bool,
                                        device=self.device)  # XOR gate
        nan_tensor = float('nan') * torch.ones_like(concordant_pairs)
        concordant_prediction = torch.where(concordant_pairs, prediction, nan_tensor)
        discordant_prediction = torch.where(discordant_pairs, prediction, nan_tensor)
        return concordant_prediction, discordant_prediction

    def run_prediction(self, batch):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
        if self.config.log_tmrca:
            output['output'] = torch.exp(output['output'])
        if self.config.constant_piecewise_output:
            output['output_const_' + str(self.const_threshold)] = self.get_const_piecewise(copy.deepcopy(output),
                                                                                           self.const_threshold)
            return output['output_const_' + str(self.const_threshold)]
        else:
            return output['output']

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

    def restore_session(self, restore_path):

        config_path = os.path.join(restore_path, 'config.yml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

        self.restore_session_name = self.config.session_name
        self.config.update_config(self.config_run)

        self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')

    def create_dataset_name(self):

        const_threshold = ''
        if 'const_threshold' in self.config_run:
            const_threshold = '.const_threshold.' + str(self.config_run['const_threshold'])

        dataset_name = 'CHR.' + str(self.config_run['chr']) \
                       + const_threshold

        return dataset_name

    def create_session_name(self):

        dataset_name = self.create_dataset_name()
        now = datetime.datetime.now()
        session_name = '{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_session_{}_{}'.format(
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            socket.gethostname(),
            dataset_name)
        session_name = os.path.join(self.config_run['output_path'], session_name)
        os.makedirs(session_name)
        return session_name

    def initialise_session(self):
        # config run file
        config_run_file = os.path.join(self.options.config)
        with open(config_run_file) as f:
            self.config_run = yaml.safe_load(f)
        if self.options.const_threshold:
            self.config_run['constant_piecewise_output'] = True
            self.config_run['const_threshold'] = float(self.options.const_threshold)

        if self.options.batch_size:
            self.config_run['batch_size'] = self.options.batch_size
        if self.options.demo:
            self.config_run['demo'] = self.options.demo
        if self.options.restore_path:
            self.config_run['restore_path'] = self.options.restore_path
        if self.options.output_path:
            self.config_run['output_path'] = self.options.output_path

        self.session_name = self.create_session_name()
        self.config_run['session_name'] = self.session_name
        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))

        # Copy config run file to output_path
        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(self.config_run, f)
