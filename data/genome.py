import torch
import numpy as np
from torch.utils.data import Dataset
from data.msprime import simulate
import os
from data.random_crop import RandomCrop
import random
import copy
from multiprocessing import Pool
import gzip
import subprocess
import math
from data.sampler import ThreadingSampler
import sys


class GenomeDataset(Dataset):
    """
    Tree sequence to genomes.
    """

    def __init__(self, config, mode, input_size=None, focus_input_size=None, context_size=None, root_time=None):

        self.config = config
        self.mode = mode
        self.feature_cm = config.feature_cm
        self.feature_bp = config.feature_bp
        self.feature_maf = config.feature_maf
        self.feature_ibs = config.feature_ibs
        self.data_type = config.data_type
        self.num_diploids, self.num_haplotypes, self.haplotypes = self.get_sizes(config)
        self.num_pairs = self.get_num_pairs(config)
        self.num_features = self.count_num_features(config)
        self.log_tmrca = config.log_tmrca
        self.eval = self.do_eval(config)
        self.Ne = self.get_Ne(config)
        self.input_size, self.focus_input_size, self.context_size = input_size, focus_input_size, context_size
        self.seeds = self.get_seeds(config)

        if hasattr(config, 'use_offset') and config.use_offset:
            self.use_offset = True
            self.num_offset_samples, self.num_modern_samples = self.get_offset_sizes(config)
        else:
            self.use_offset = False

        if mode == 'run':
            # Could not run simulations in parallel for some reason
            self.simulations = [self.run_simulation(self.seeds[0])]
        else:
            self.num_simulations = len(self.seeds)
            # Run all simulations in parallel
            if self.data_type == 'impute':
                pool = Pool(config.num_simulations)
            else:
                pool = Pool(config.n_workers)
            self.simulations = list(pool.map(self.run_simulation, self.seeds))
            pool.close()
            pool.join()

        self.get_input_size(config, input_size, focus_input_size)
        self.root_time = self.get_root_time(root_time)

        # Apply transformation if there is one
        self.transform = None
        self.initialise_transform(config)

        # Initialize threading sampler
        self.threading_sampler = None
        self.initialise_threading_sampler(config)

    def get_root_time(self, root_time):
        if root_time is None:
            root_time = 0
            for tree in self.simulations[0]['tree_sequence'].trees():
                if len(tree.roots) == 1:
                    root_time = max(tree.get_time(tree.get_root()), root_time)
            root_time = 10 ** len(str(int(root_time)))
            if self.config.log_tmrca:
                root_time = round(math.log(root_time), 2)
        return root_time

    def get_input_size(self, config, input_size, focus_input_size):

        bin_size = config.model.bin
        bin_unit = config.model.bin_unit
        num_variants_list = [simulation['num_variants'] for simulation in self.simulations]
        min_seed = num_variants_list.index(min(num_variants_list))
        num_variants = self.simulations[min_seed]['num_variants']
        phys_pos = self.simulations[min_seed]['phys_pos']
        gen_pos = self.simulations[min_seed]['gen_pos']

        if input_size is None:
            if bin_unit == 'cM':
                prev_gen_pos = np.zeros(num_variants)
                prev_gen_pos[1:] = gen_pos.reshape(-1)[:-1]
                distances = gen_pos - prev_gen_pos
                self.focus_input_size = int(bin_size / np.mean(distances))
            elif bin_unit == 'bp':
                prev_phys_pos = np.zeros(num_variants)
                prev_phys_pos[1:] = phys_pos[:-1]
                distances = phys_pos - prev_phys_pos
                self.focus_input_size = int(bin_size / np.mean(distances))
            elif bin_unit == 'variant':
                self.focus_input_size = bin_size
            self.input_size = self.focus_input_size
            if config.model.network == 'cnn':
                kernel_size = config.model.kernel_size
                dilation = len(kernel_size) * [1]
                dilation[0] = math.ceil(self.focus_input_size / (2 * kernel_size[0]))
                for i in range(len(kernel_size)):
                    self.input_size += dilation[i] * (kernel_size[i] - 1)

        else:
            self.input_size = input_size
            self.focus_input_size = focus_input_size

        # context_size is always an integer if kernels are odds
        self.context_size = int((self.input_size - self.focus_input_size) / 2)

    def initialise_threading_sampler(self, config):
        if self.mode == 'train' and config.threading_sampler:
            self.threading_sampler = ThreadingSampler(self.simulations, self.num_haplotypes, config)

    def __len__(self):
        # return self.num_pairs
        if self.mode == 'train' and self.threading_sampler:
            return self.num_simulations * self.num_haplotypes
        else:
            return self.num_pairs

    def __getitem__(self, index):
        if self.threading_sampler:
            seed_idx, haplotype_i, haplotype_j = self.threading_sampler.sample(index)
            simulation = self.simulations[seed_idx]
        else:
            # randomly select a seed to sample from
            seed_idx = random.randrange(0, len(self.seeds))
            simulation = self.simulations[seed_idx]
            (haplotype_i, haplotype_j) = simulation['pair_to_haplotypes'][index]

        num_variants = simulation['num_variants']

        batch = dict()

        start_label, end_label = self.construct_batch_input(haplotype_i, haplotype_j,
                                                            simulation, seed_idx, num_variants, batch)

        if self.eval:
            # for debugging purposes
            if start_label >= min(end_label, num_variants):
                print('start_label', start_label, 'end_label', end_label,
                      'num_variants', num_variants, 'seed_idx', seed_idx)
            self.construct_batch_label(haplotype_i, haplotype_j, simulation,
                                       num_variants, batch, start_label, end_label)

        return batch

    def construct_batch_input(self, haplotype_i, haplotype_j, simulation, seed_idx, num_variants, batch):

        input = self.get_input(haplotype_i, haplotype_j, simulation, seed_idx)

        if self.transform:
            if num_variants < self.context_size:
                # the input is too small, we need to pad it (on the left side)
                missing_variants = self.input_size - num_variants
                input = np.pad(input, ((0, 0), (missing_variants, 0)), 'constant')
                batch['input'] = torch.from_numpy(input)
                start_label = 0
                end_label = start_label + self.focus_input_size
            elif num_variants < self.input_size:
                # the input is too small, we need to pad it (on the right side, arbitrary choice)
                missing_variants = self.input_size - num_variants
                input = np.pad(input, ((0, 0), (0, missing_variants)), 'constant')
                batch['input'] = torch.from_numpy(input)
                start_label = self.context_size
                end_label = start_label + self.focus_input_size
            else:
                # apply transform
                start_transform, end_transform = self.transform()  # start_transform >= -context_size
                if start_transform < 0:
                    input = np.pad(input, ((0, 0), (abs(start_transform), 0)), 'constant')
                    start = 0
                else:
                    start = start_transform
                if end_transform > num_variants:
                    missing_variants = end_transform - num_variants
                    input = np.pad(input, ((0, 0), (0, missing_variants)), 'constant')
                batch['input'] = torch.from_numpy(input)[:, start:start + self.input_size]
                start_label = start_transform + self.context_size  # always >= 0 and <= num_variants - focus_input_size
                end_label = start_label + self.focus_input_size  # always >= focus_input_size and <= num_variants
        else:
            # no transform
            input = np.pad(input, ((0, 0), (self.context_size, self.context_size)), 'constant')
            batch['input'] = torch.from_numpy(input)
            start_label = 0
            end_label = num_variants

        return start_label, end_label

    def construct_batch_label(self, haplotype_i, haplotype_j, simulation,
                              num_variants, batch, start_label, end_label):

        label = self.get_label(haplotype_i, haplotype_j, start_label, min(end_label, num_variants), simulation)
        breakpoints = self.get_recombination_points(label)
        pos_index = np.asarray(range(start_label, min(end_label, num_variants)))

        if self.transform and end_label > num_variants:
            # the input is too small, we need to pad it (on the right side, arbitrary choice)
            missing_variants = end_label - num_variants
            label = np.pad(label, (0, missing_variants), 'edge')
            breakpoints = np.pad(breakpoints, (0, missing_variants), 'constant')
            pos_index = np.pad(pos_index, (0, missing_variants), 'constant')

        batch['label'] = torch.from_numpy(label)
        batch['breakpoints'] = torch.from_numpy(breakpoints)
        batch['pos_index'] = torch.from_numpy(pos_index)

    def get_Ne(self, config):
        if self.mode == 'train':
            Ne = config.num_simulations * [config.Ne]
        else:
            Ne = [config.Ne]
        return Ne

    def get_num_pairs(self, config):
        if self.mode == 'train':
            return int(((self.num_haplotypes * (self.num_haplotypes - 1)) / 2) + self.num_haplotypes)
        else:
            return int((self.num_haplotypes * (self.num_haplotypes - 1)) / 2)

    @staticmethod
    def filter_vcf(simulation, diploids, tot_num_diploids, tag=None, variants=None):

        if variants is None:
            variants = np.ones(len(simulation['phys_pos']), dtype=bool)

        vcf_file = gzip.open(simulation['path_dataset'] + ".vcf.gz", "rt")
        vcf_filter_file = gzip.open(simulation['path_dataset'] + tag + ".vcf.gz", "wt")
        for i, vcf_line in enumerate(vcf_file):

            # vcf_line = vcf_line.decode('ascii')

            if i <= 4:
                # copy header
                vcf_filter_file.write(vcf_line)

            elif i == 5:
                vcf_line = vcf_line.split()
                vcf_filter_string = '\t'.join(vcf_line[:9])
                for j in range(tot_num_diploids):
                    if j in diploids:
                        vcf_filter_string += '\t' + str(vcf_line[j + 9])
                vcf_filter_file.write(vcf_filter_string + '\n')

            else:
                if variants[i - 6]:
                    vcf_line = vcf_line.split()
                    # issue with tskit 0.3 for REF and ALT field
                    # need to use phys_pos to avoid replicates
                    vcf_filter_string = 'chr' + str(simulation['chromosome']) + '\t' \
                                        + str(simulation['phys_pos'][i - 6]) + '\t' \
                                        + vcf_line[2] + '\t' \
                                        + 'A\tT\t' + '\t'.join(vcf_line[5:9])

                    for j in range(tot_num_diploids):
                        if j in diploids:
                            vcf_filter_string += '\t' + str(vcf_line[j + 9])
                    vcf_filter_file.write(vcf_filter_string + '\n')

        vcf_file.close()
        vcf_filter_file.close()

        return

    @staticmethod
    def impute_genotype_matrix(simulation):
        diploids_target = simulation['diploids_target']
        dr2 = np.zeros(simulation['genotype_matrix'].shape[1])

        with gzip.open(simulation['path_dataset'] + ".output.vcf.gz", "rt") as vcf_file:
            for i, vcf_line in enumerate(vcf_file):
                # skip header
                if i >= 11:
                    variant = i - 11
                    # vcf_line = vcf_line.decode('ascii').split()
                    vcf_line = vcf_line.split()
                    info = vcf_line[7]
                    dr2[variant] = float(info.replace("=", ";").split(';')[1])

                    if 'IMP' in info:
                        for j in range(len(diploids_target)):
                            imputed_data = vcf_line[9 + j]  # imputed data of jth individual
                            imputed_data = imputed_data.replace("|", ":").split(':')
                            imputed_data = [float(i) for i in imputed_data]
                            hap_j = [imputed_data[-2], imputed_data[-1]]
                            id_hap_j = [2 * diploids_target[j], 2 * diploids_target[j] + 1]
                            simulation['genotype_matrix'][id_hap_j[0], variant] = hap_j[0]
                            simulation['genotype_matrix'][id_hap_j[1], variant] = hap_j[1]

            simulation['dr2'] = dr2.reshape(1, -1)

        return simulation

    @staticmethod
    def write_map(simulation):
        path = simulation['path_dataset']
        phys_pos = simulation['phys_pos']
        gen_pos = simulation['gen_pos']
        chromosome = simulation['chromosome']
        with open(path + '.map', 'w') as out_file:
            for i in range(0, len(phys_pos)):
                out_file.write('\t'.join(['chr' + str(chromosome),
                                          'SNP_' + str(phys_pos[i]),
                                          str(gen_pos[i]),
                                          str(phys_pos[i])]) + '\n')
        return

    def impute(self, config, simulation, num_diploids_ref):

        num_haplotypes = self.num_haplotypes + 2 * num_diploids_ref
        num_diploids = self.num_diploids + num_diploids_ref

        # make vcf
        with gzip.open(simulation['path_dataset'] + ".vcf.gz", "wt") as vcf_file:
            simulation['tree_sequence'].write_vcf(vcf_file)

        # prepare downsample variants
        random_samples = np.random.choice(range(num_haplotypes), size=2 * config.downsample_size, replace=False)
        variants = simulation['genotype_matrix'][random_samples, :].any(axis=0)

        # filter vcf to build vcf_ref and vcf_target_array
        if self.mode == 'val':
            # remove randomness
            diploids_ref = np.arange(num_diploids_ref)
        else:
            diploids_ref = np.sort(np.random.choice(range(num_diploids), size=num_diploids_ref, replace=False))
        diploids_target = [i for i in range(num_diploids) if i not in diploids_ref]
        self.filter_vcf(simulation, diploids_ref, num_diploids, tag='.ref', variants=variants)
        self.filter_vcf(simulation, diploids_target, num_diploids, tag='.target_array',
                        variants=simulation['filter_array'])
        self.write_map(simulation)

        # downsample variants in simulation
        simulation = self.downsample_variants(self.config, simulation, variants=variants)

        # beagle impute
        subprocess.run(['java', '-jar', '../simulation_impute_pipeline/beagle.25Nov19.28d.jar',
                        "ref=" + simulation['path_dataset'] + ".ref.vcf.gz",
                        "gt=" + simulation['path_dataset'] + ".target_array.vcf.gz",
                        'ap=true',
                        "map=" + simulation['path_dataset'] + ".map",
                        "out=" + simulation['path_dataset'] + ".output"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL
                       )

        # impute genotype_matrix
        simulation['diploids_target'] = diploids_target
        simulation['diploids_ref'] = diploids_ref
        simulation = self.impute_genotype_matrix(simulation)

        # delete files
        os.remove(simulation['path_dataset'] + ".vcf.gz")
        os.remove(simulation['path_dataset'] + ".ref.vcf.gz")
        os.remove(simulation['path_dataset'] + ".target_array.vcf.gz")
        os.remove(simulation['path_dataset'] + ".output.vcf.gz")
        os.remove(simulation['path_dataset'] + ".output.log")
        os.remove(simulation['path_dataset'] + ".map")

        return simulation

    def run_simulation(self, seed):
        """Run simulation for a given pair of seed and seed index."""

        if self.data_type == 'array':
            simulation = self.build_dataset(self.num_diploids, self.config, seed)
            # Filter maf in array mode
            simulation = self.filter_attributes(simulation)
        elif self.data_type == 'sequence':
            simulation = self.build_dataset(self.num_diploids, self.config, seed)
            # Downsample variants so that simulation is independent of sample size in sequence mode
            if self.config.downsample_size != self.num_diploids:
                simulation = self.downsample_variants(self.config, simulation)
        elif self.data_type == 'impute':
            if self.mode == 'train':
                num_diploids_ref = random.randint(150, 1000)
            else:
                num_diploids_ref = self.config.val_ref_size
            simulation = self.build_dataset(self.num_diploids + num_diploids_ref, self.config, seed)
            simulation = self.impute(self.config, simulation, num_diploids_ref)

        simulation['num_variants'] = self.get_num_variants(self.mode, simulation['phys_pos'], self.input_size)
        simulation['maf'] = simulation['maf'].reshape(1, -1)
        simulation['gen_dist'] = self.get_gen_dist(simulation['gen_pos']).reshape(1, -1)
        simulation['phys_dist'] = self.get_phys_dist(simulation['phys_pos']).reshape(1, -1)
        # simulation['gen_pos'] = simulation['gen_pos'].reshape(1, -1)
        # simulation['phys_pos'] = simulation['phys_pos'].reshape(1, -1)
        simulation['pair_to_haplotypes'], simulation['haplotypes_to_pair'] = self.get_pair_haplotypes(simulation)

        return simulation

    def do_eval(self, config):
        evaluation = True
        if self.mode == 'run':
            if not config.eval:
                evaluation = False
        return evaluation

    def get_seeds(self, config):
        """Get relevant seeds for the class."""
        if self.mode == 'train':
            seeds = random.sample(range(config.seed_val + 1, 2 ** 8), k=config.num_simulations)
        elif self.mode == 'val':
            seeds = [config.seed_val]
        elif self.mode == 'run':
            seeds = [config.seed_run]
        return seeds

    def initialise_transform(self, config):
        """Initialise the transformation to be performed on data."""

        num_variants = [simulation['num_variants'] for simulation in self.simulations]
        min_seed = num_variants.index(min(num_variants))

        if self.mode == 'run':
            self.transform = None

        elif hasattr(config, 'transform'):
            if config.transform == 'random_crop':
                self.transform = RandomCrop(self.simulations[min_seed]['num_variants'],
                                            self.input_size,
                                            self.context_size)
                print('Random crop transform has been set up with max start index', self.transform.max_start_index)

        else:
            self.transform = None

    @staticmethod
    def filter_attributes(simulation):
        """Filter SNPs in attributes to get SNP array data."""
        variants = simulation['filter_array']
        simulation['genotype_matrix'] = simulation['genotype_matrix'][:, variants]
        simulation['phys_pos'] = simulation['phys_pos'][variants]
        simulation['phys_pos_float'] = simulation['phys_pos_float'][variants]
        simulation['gen_pos'] = simulation['gen_pos'][variants]
        simulation['maf'] = simulation['maf'][variants]
        simulation['mutations_age'] = simulation['mutations_age'][variants]
        return simulation

    def downsample_variants(self, config, simulation, variants=None):
        """Downsample variants so that genomic distances do not depend on sample size."""
        if variants is None:
            random_samples = np.random.choice(range(self.num_haplotypes), size=2 * config.downsample_size,
                                              replace=False)
            variants = simulation['genotype_matrix'][random_samples, :].any(axis=0)
        simulation['genotype_matrix'] = simulation['genotype_matrix'][:, variants]
        simulation['phys_pos'] = simulation['phys_pos'][variants]
        simulation['phys_pos_float'] = simulation['phys_pos_float'][variants]
        simulation['gen_pos'] = simulation['gen_pos'][variants]
        simulation['maf'] = simulation['maf'][variants]
        simulation['mutations_age'] = simulation['mutations_age'][variants]
        # simulation['filter_array'] = np.logical_and(variants, simulation['filter_array'])
        simulation['downsample'] = variants
        return simulation

    @staticmethod
    def read_genotypes_pair(haplotype_i, haplotype_j, simulation):
        """Read input array for a given pair."""
        return simulation['genotype_matrix'][[haplotype_i, haplotype_j]].astype('float32')

    @staticmethod
    def create_gntp_errors(genotype_matrix, phys_pos, error_rate):
        num_errors = int(error_rate * len(phys_pos))
        for i in range(genotype_matrix.shape[0]):
            error_positions = np.random.choice(len(phys_pos), num_errors, replace=False)
            genotype_matrix[i, error_positions] = 1 - genotype_matrix[i, error_positions]
        return genotype_matrix

    @staticmethod
    def create_phasing_error(genotype_matrix, switch_error_rate):

        num_haplotypes = genotype_matrix.shape[0]
        switch_segments = {}

        for haplotype in range(0, num_haplotypes, 2):
            # process one diploid individual at a time
            diploid_label = (int(haplotype / 2))
            diplotype = genotype_matrix[[haplotype, haplotype + 1]].astype('float32')
            new_diplotype = copy.deepcopy(diplotype)
            composite = np.sum(diplotype, axis=0)
            het_sites = np.where(composite == 1)[0]

            num_switch_errors = int(switch_error_rate * len(het_sites))
            # make num_switch_errors even, this will facilitate the segments representation
            num_switch_errors = int(num_switch_errors) if ((int(num_switch_errors) % 2) == 0) else int(
                num_switch_errors) + 1
            switch_error_positions = np.sort(np.random.choice(het_sites, num_switch_errors, replace=False))

            segments = np.array_split(switch_error_positions, num_switch_errors / 2)
            segments = np.stack(segments, axis=0)  # should now be (num_switch_errors/2, 2)-shaped array

            i = 0
            while i < num_switch_errors - 1:
                if i % 2 == 0:
                    new_diplotype[0, switch_error_positions[i]:switch_error_positions[i + 1]] = \
                        diplotype[1, switch_error_positions[i]:switch_error_positions[i + 1]]
                    new_diplotype[1, switch_error_positions[i]:switch_error_positions[i + 1]] = \
                        diplotype[0, switch_error_positions[i]:switch_error_positions[i + 1]]

                i += 1

            genotype_matrix[[haplotype, haplotype + 1]] = new_diplotype
            switch_segments[diploid_label] = segments

        return genotype_matrix, switch_segments

    @staticmethod
    def island_props(v):
        # from https://stackoverflow.com/questions/46502265/find-constant-subarrays-in-large-numpy-array
        # Get one-off shifted slices and then compare element-wise, to give
        # us a mask of start and start positions for each island.
        # Also, get the corresponding indices.
        mask = np.concatenate(([True], v[1:] != v[:-1], [True]))
        loc0 = np.flatnonzero(mask)

        # Get the start locations
        loc = loc0[:-1]

        # The values would be input array index by the start locations.
        # The lengths would be the differentiation between start and stop indices.
        return v[loc], loc, np.diff(loc0)

    @staticmethod
    def compute_ibs(xor_array):
        values, loc, lengths = GenomeDataset.island_props(xor_array)
        end_ix = 0
        ibs = np.zeros_like(xor_array)
        for i in range(len(lengths)):
            start_idx = loc[i]
            end_ix += lengths[i]
            if values[i] == 0:
                ibs[start_idx:end_ix] = lengths[i]
        return ibs.reshape(1, -1)

    def get_input(self, haplotype_i, haplotype_j, simulation, seed_idx):
        """Get input for a given pair."""
        genotypes = self.read_genotypes_pair(haplotype_i, haplotype_j, simulation)
        input = np.zeros(genotypes.shape).astype('float32')

        np.set_printoptions(threshold=sys.maxsize)

        if self.data_type == 'impute':
            input[0, :] = genotypes[0, :]
            input[1, :] = genotypes[1, :]
        else:
            input[0, :] = np.logical_xor(genotypes[0, :], genotypes[1, :]).astype(float)
            input[1, :] = np.logical_and(genotypes[0, :], genotypes[1, :]).astype(float)

        if self.feature_cm:
            input = np.concatenate((input, simulation['gen_dist']))
            # input = np.concatenate((input, simulation['gen_pos']))
        if self.feature_bp:
            input = np.concatenate((input, simulation['phys_dist']))
            # input = np.concatenate((input, simulation['phys_pos']))
        if self.feature_maf:
            input = np.concatenate((input, simulation['maf']))
        if self.feature_ibs:
            if self.data_type == 'impute':
                xor_array = np.logical_xor(genotypes[0, :], genotypes[1, :]).astype(float)
            else:
                xor_array = input[0, :]
            ibs = self.compute_ibs(xor_array)
            input = np.concatenate((input, ibs))

        # if self.feature_demo_const:
        #    Ne = self.Ne[seed_idx] * np.ones_like(simulation['gen_dist'])
        #    input = np.concatenate((input, Ne))
        # if self.data_type == 'impute':
        #    input = np.concatenate((input, simulation['dr2'])).astype('float32')

        return input.astype('float32')

    def read_tmrca_pair(self, haplotype_i, haplotype_j, seed):
        """Read label array for a given pair and a given seed."""

        label = np.zeros((1, self.simulations[seed]['num_variants']))
        pos_index = 0

        for tree in self.simulations[seed]['tree_sequence'].trees():
            tree_end = tree.interval[1]
            # tmrca = tree.time(tree.mrca(haplotype_i, haplotype_j))
            tmrca = tree.get_tmrca(haplotype_i, haplotype_j)
            while pos_index < self.simulations[seed]['num_variants'] and self.simulations[seed]['phys_pos'][
                pos_index] <= tree_end:
                label[0, pos_index] = tmrca
                pos_index += 1

        return label

    @staticmethod
    def read_ground_truth_pair(haplotype_i, haplotype_j, start, end, simulation):
        """Read label array for a given pair from msprime simulation."""

        num_variants = end - start
        label = np.ones(num_variants).astype('float32')

        if haplotype_i == haplotype_j:
            return 0.1 * label

        else:
            pos_index = 0
            cur_pos = start

            def is_in_tree(i, tree_start_pos, tree_end_pos):
                """Return true if position index i is in tree."""
                pos = simulation['phys_pos_float'][i]
                return tree_start_pos <= pos <= tree_end_pos

            for tree in simulation['tree_sequence'].trees():
                tree_start, tree_end = tree.interval

                if cur_pos == end:
                    # we are done, stop now
                    break

                if is_in_tree(cur_pos, tree_start, tree_end):
                    # tmrca = tree.time(tree.mrca(haplotype_i, haplotype_j))
                    tmrca = tree.get_tmrca(haplotype_i, haplotype_j)
                    while pos_index < num_variants and is_in_tree(cur_pos, tree_start, tree_end):
                        label[pos_index] = tmrca
                        pos_index += 1
                        cur_pos = start + pos_index

                if tree_start > simulation['phys_pos_float'][end - 1]:
                    # no need to keep looping, stop now
                    break

            assert pos_index == num_variants

            return label

    def get_label(self, haplotype_i, haplotype_j, start, end, simulation):
        """Get label for a given pair."""
        if simulation['switch_error_segments'] is not None:
            label = self.read_ground_truth_pair_with_phasing_error(haplotype_i, haplotype_j, start, end, simulation)
        else:
            label = self.read_ground_truth_pair(haplotype_i, haplotype_j, start, end, simulation)
        if self.log_tmrca:
            label = np.log(label)
        return label

    def read_ground_truth_pair_with_phasing_error(self, haplotype_i, haplotype_j, start, end, simulation):

        if haplotype_i % 2 == 0:
            diploid_i = int(haplotype_i / 2)
            haplotype_i_2 = haplotype_i + 1
        else:
            diploid_i = int((haplotype_i - 1) / 2)
            haplotype_i_2 = haplotype_i - 1

        if haplotype_j % 2 == 0:
            diploid_j = int(haplotype_j / 2)
            haplotype_j_2 = haplotype_j + 1
        else:
            diploid_j = int((haplotype_j - 1) / 2)
            haplotype_j_2 = haplotype_j - 1

        label = self.read_ground_truth_pair(haplotype_i, haplotype_j, start, end, simulation)
        label_i2_j = self.read_ground_truth_pair(haplotype_i_2, haplotype_j, start, end, simulation)
        label_i_j2 = self.read_ground_truth_pair(haplotype_i, haplotype_j_2, start, end, simulation)
        label_i2_j2 = self.read_ground_truth_pair(haplotype_i_2, haplotype_j_2, start, end, simulation)

        # get switch_error_segments and rescale them to labels indexes
        switch_segments_i = simulation['switch_error_segments'][diploid_i] - start
        switch_segments_j = simulation['switch_error_segments'][diploid_j] - start

        num_swaps = np.zeros_like(label)

        # swap label and label_i2_j
        for start_segment, end_segment in switch_segments_i:
            if start_segment >= 0:
                label[start_segment: end_segment] = label_i2_j[start_segment: end_segment]
                num_swaps[start_segment: end_segment] += 1
            elif start_segment < 0 < end_segment:
                label[:end_segment] = label_i2_j[:end_segment]
                num_swaps[:end_segment] += 1

        # swap label and label_i_j2
        for start_segment, end_segment in switch_segments_j:
            if start_segment >= 0:
                label[start_segment: end_segment] = label_i_j2[start_segment: end_segment]
                num_swaps[start_segment: end_segment] += 1
            elif start_segment < 0 < end_segment:
                label[:end_segment] = label_i2_j[:end_segment]
                num_swaps[:end_segment] += 1

        # swap label and label_i2_j2 if there is overlap
        label[num_swaps == 2] = label_i2_j2[num_swaps == 2]

        return label

    def get_pair_haplotypes(self, simulation):
        """Get mapping between pair index and haplotypes."""
        pair_to_haplotypes = {}
        haplotypes_to_pair = {}
        cpt_pair = 0

        if self.data_type == 'impute':
            diploids_target = simulation['diploids_target']
            for i, diploid_i in enumerate(diploids_target):
                for j, diploid_j in enumerate(diploids_target[:i]):
                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i, 2 * diploid_j)
                    haplotypes_to_pair[(2 * diploid_i, 2 * diploid_j)] = cpt_pair
                    cpt_pair += 1

                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i, 2 * diploid_j + 1)
                    haplotypes_to_pair[(2 * diploid_i, 2 * diploid_j + 1)] = cpt_pair
                    cpt_pair += 1

                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i + 1, 2 * diploid_j)
                    haplotypes_to_pair[(2 * diploid_i + 1, 2 * diploid_j)] = cpt_pair
                    cpt_pair += 1

                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i + 1, 2 * diploid_j + 1)
                    haplotypes_to_pair[(2 * diploid_i + 1, 2 * diploid_j + 1)] = cpt_pair
                    cpt_pair += 1

                pair_to_haplotypes[cpt_pair] = (2 * diploid_i + 1, 2 * diploid_i)
                haplotypes_to_pair[(2 * diploid_i + 1, 2 * diploid_i)] = cpt_pair
                cpt_pair += 1

                if self.mode == 'train':
                    # add within pairs
                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i, 2 * diploid_i)
                    haplotypes_to_pair[(2 * diploid_i, 2 * diploid_i)] = cpt_pair
                    cpt_pair += 1
                    pair_to_haplotypes[cpt_pair] = (2 * diploid_i + 1, 2 * diploid_i + 1)
                    haplotypes_to_pair[(2 * diploid_i + 1, 2 * diploid_i + 1)] = cpt_pair
                    cpt_pair += 1

        else:
            for haplotype_i in self.haplotypes:
                if self.mode == 'train':
                    # add within pairs
                    haplotype_j_end = haplotype_i + 1
                else:
                    haplotype_j_end = haplotype_i
                for haplotype_j in range(haplotype_j_end):
                    pair_to_haplotypes[cpt_pair] = (haplotype_i, haplotype_j)
                    haplotypes_to_pair[(haplotype_i, haplotype_j)] = cpt_pair
                    cpt_pair += 1

        assert cpt_pair == self.num_pairs, "Total number of pairs needs to be identical to the dataset size."

        return pair_to_haplotypes, haplotypes_to_pair

    def get_sizes(self, config):
        """Get useful parameters for the dataset."""

        if self.mode == 'train':
            num_diploids = config.sample_size_train
        elif self.mode == 'val':
            num_diploids = config.sample_size_val
        elif self.mode == 'run':
            num_diploids = config.sample_size_run

        num_haplotypes = 2 * num_diploids
        haplotypes = range(0, num_haplotypes)
        assert num_haplotypes >= 2 * config.downsample_size, "Sample size needs to be bigger than downsampling size."

        return num_diploids, num_haplotypes, haplotypes

    def get_offset_sizes(self, config):
        """Get useful values for the dataset if there is a sampling offset."""
        if not hasattr(config, 'sample_size_offset'):
            raise ValueError('Need to specify the number of samples with offset')
        elif not hasattr(config, 'offset_range'):
            raise ValueError('Need to specify the range of possible offsets')
        else:
            num_offset_samples = config.sample_size_offset
            num_modern_samples = self.num_diploids - num_offset_samples
        return num_offset_samples, num_modern_samples

    @staticmethod
    def get_rec_map_name(chromosome, chromosome_region, demo, reference_genome):
        rec_map = ''
        if reference_genome == 'hg19':
            if demo == 'constant' or demo == 'CEU.Terhorst' or demo=='bottleneck': #TODO:remove bottleneck
                # use the average CEU_YRI_ASN rec map for constant pop size and CEU.Terhorst
                path_demo = './files/genetic-maps-shifted-no-centromere/average_CEU_YRI_ASN/average_CEU_YRI_ASN'
            else:
                # path_demo = './files/genetic-maps/' + demo + '/' + demo
                path_demo = './files/genetic-maps-shifted-no-centromere/' + demo + '/' + demo
            if chromosome in [13, 14, 15, 21, 22]:
            # no centromere
                rec_map = path_demo \
                        + '_recombination_map_hapmap_format_' + reference_genome + '_chr_' \
                        + str(chromosome) \
                        + '.txt.gz'
            else:
                rec_map = path_demo \
                        + '_recombination_map_hapmap_format_' + reference_genome + '_chr_' \
                        + str(chromosome) \
                        + '.' + str(chromosome_region) \
                        + '.txt.gz'

        elif reference_genome == 'hg38':
            if demo == 'constant':
                # use the CEU recombination MAP for constant pop size
                path_demo = './files/genetic-maps-hg38/CEU/CEU'
            else:
                path_demo = './files/genetic-maps-hg38/' + demo + '/' + demo
            rec_map = path_demo \
                      + '_recombination_map_hapmap_format_' + reference_genome + '_chr_' \
                      + str(chromosome) \
                      + '.txt.gz'
        return rec_map

    @staticmethod
    def get_chromosomes(mode, config):
        # Random chromosomal region
        if mode == 'train':
            chromosome = random.choice(range(1, 23))
            chromosome_region = random.choice(range(1, 3))
        elif mode == 'val':
            chromosome = 2
            chromosome_region = 1
        elif mode == 'run':
            chromosome = config.chr
            chromosome_region = 1
        # print("Simulating chromosome " + str(chromosome)
        #      + "." + str(chromosome_region)
        #      + " with seed " + str(seed) + "...")
        return chromosome, chromosome_region

    def build_dataset(self, num_diploids, config, seed):
        """Build dataset."""

        # Create samples dictionary if using offset
        if self.use_offset:
            sample_dict = self.get_sample_dict(config)
        else:
            sample_dict = None

        demo = config.demography
        chromosome, chromosome_region = self.get_chromosomes(self.mode, config)

        # Creating path to dataset
        dataset_name = self.create_dataset_name(config, self.mode,
                                                chromosome, seed)
        path_dataset = os.path.join(config.session_name, dataset_name)

        # Read genetic positions
        rec_map = self.get_rec_map_name(chromosome, chromosome_region, demo, config.reference_genome)
        genetic_map = self.read_genetic_map(rec_map)

        # Crop rec and gen maps
        # if self.mode == 'train' or self.mode == 'val' or self.mode == 'run':
        genetic_map = self.crop_genetic_map(genetic_map, self.mode)
        rec_map = self.crop_rec_map(config, chromosome, dataset_name, genetic_map)

        # Simulate
        index_seed = self.seeds.index(seed)
        Ne = self.Ne[index_seed]
        tree_sequence = simulate(num_diploids, config, seed, rec_map, Ne, sample_dict)
        genotype_matrix = np.transpose(tree_sequence.genotype_matrix()).astype('float32')
        phys_pos_float = np.array([variant.site.position
                                   for variant in tree_sequence.variants()])
        phys_pos = phys_pos_float.astype('int')
        phys_pos = self.remove_duplicates(phys_pos)
        gen_pos = self.get_genetic_positions(phys_pos, genetic_map, config)

        # Create genotyping errors
        if hasattr(config, 'gntp_error_rate') and config.gntp_error_rate != 0:
            genotype_matrix = self.create_gntp_errors(genotype_matrix, phys_pos, config.gntp_error_rate)

        # Create phasing errors
        switch_error_segments = None
        if hasattr(self.config, 'switch_error_rate') and self.config.switch_error_rate != 0:
            genotype_matrix, switch_error_segments = self.create_phasing_error(genotype_matrix,
                                                                               config.switch_error_rate)

        # Compute maf
        maf = np.sum(genotype_matrix, axis=0) / genotype_matrix.shape[0]
        maf = np.where(maf < 0.5, maf, 1 - maf)

        # Get SNP array data
        filter_array = self.make_array(maf, chromosome, rec_map, demo, config.reference_genome)

        # Get mutations age
        mutations_age = self.get_mutations_age(tree_sequence, phys_pos_float)

        # Delete temporary rec map file
        if os.path.exists(rec_map):
            os.remove(rec_map)

        simulation = {'tree_sequence': tree_sequence,
                      'genotype_matrix': genotype_matrix,
                      'phys_pos_float': phys_pos_float,
                      'phys_pos': phys_pos,
                      'gen_pos': gen_pos,
                      'maf': maf,
                      'mutations_age': mutations_age,
                      'filter_array': filter_array,
                      'path_dataset': path_dataset,
                      'dataset_name': dataset_name,
                      'chromosome': chromosome,
                      'switch_error_segments': switch_error_segments}

        return simulation

    def get_sample_dict(self, config):
        ancestor_from = config.offset_range[0]
        ancestor_to = config.offset_range[1]
        if self.mode == 'val':
            np.random.seed(0)

        sample_dict = dict()
        # Sampling times for offset individuals must be provided in GENERATIONS
        ancient_sampling_times = np.random.choice(np.arange(ancestor_from, ancestor_to + 1, dtype='int'),
                                                  size=self.num_offset_samples, replace=True)

        assert len(ancient_sampling_times) == self.num_offset_samples
        sample_dict['A'] = [self.num_modern_samples, 0]
        values, counts = np.unique(ancient_sampling_times, return_counts=True)
        for idx, t in enumerate(values):
            sample_dict['A_anc' + str(t)] = [counts[idx], t]

        return sample_dict

    @staticmethod
    def get_mutations_age(tree_sequence, phys_pos):
        mutations_age = tree_sequence.tables.mutations.time
        assert len(mutations_age) == len(phys_pos), 'Error while computing min mutations age.'
        return mutations_age

    @staticmethod
    def get_recombination_points(label):
        """Get recombination points from an array of TMRCAs."""
        label_prev = np.zeros(label.shape)
        label_prev[0] = label[0]
        label_prev[1:] = label[:-1]
        breakpoints = (label - label_prev != 0).astype('int')
        # breakpoints = (label - label_prev != 0).astype('float32')
        return breakpoints

    @staticmethod
    def get_gen_dist(gen_pos):
        """Compute genetic distances from an array of genetic positions."""
        prev_gen_pos = np.zeros(len(gen_pos))
        prev_gen_pos[1:] = gen_pos[:-1]
        distances = gen_pos - prev_gen_pos
        return distances

    @staticmethod
    def get_phys_dist(phys_pos):
        """Compute physical distances from an array of physical positions."""
        prev_phys_pos = np.zeros(len(phys_pos))
        prev_phys_pos[1:] = phys_pos[:-1]
        distances = phys_pos - prev_phys_pos
        return distances

    @staticmethod
    def get_num_variants(mode, phys_pos, input_size=None):
        """Compute total number of variants."""
        num_variants = len(phys_pos)
        if mode == 'train':
            if num_variants < input_size:
                print('One simulation does not contain enough variants, inputs from it will be padded (' +
                      str(num_variants) + ' variants).')
        if mode == 'val':
            print('Validation dataset has {} variants in total.'.format(num_variants))
        elif mode == 'run':
            print('Dataset has {} variants in total.'.format(num_variants))

        return num_variants

    @staticmethod
    def make_array(maf, chromosome, rec_map, demo, reference_genome):
        """Downsample SNPs to match UKBB frequencies."""

        def compute_chr_length(file_name):
            with gzip.open(file_name, 'rt') as file_map:
                lines_map = file_map.readlines()
                start_pos = int(lines_map[1].split()[1])  # read first position from the second row
                end_pos = int(lines_map[-1].split()[1])  # read last position from the last row
                length = start_pos - end_pos
            return length

        def compute_length_factor(chromosome, rec_map, demo, reference_genome):
            if reference_genome == 'hg19':
                if demo == 'constant' or demo == 'CEU.Terhorst':
                    # use the average CEU_YRI_ASN rec map for constant pop size and CEU.Terhorst
                    path_demo = './files/genetic-maps/average_CEU_YRI_ASN/average_CEU_YRI_ASN'
                else:
                    path_demo = './files/genetic-maps/' + demo + '/' + demo
            elif reference_genome == 'hg38':
                if demo == 'constant':
                    # use the CEU recombination MAP for constant pop size
                    path_demo = './files/genetic-maps-hg38/CEU/CEU'
                else:
                    path_demo = './files/genetic-maps-hg38/' + demo + '/' + demo
            file_name = path_demo \
                        + '_recombination_map_hapmap_format_' + reference_genome + '_chr_' \
                        + str(chromosome) \
                        + '.txt.gz'
            total_length = compute_chr_length(file_name)
            length = compute_chr_length(rec_map)
            factor = length / total_length
            assert factor <= 1, 'Length factor is bigger than 1'
            return factor

        # np.random.seed(0)
        length_factor = compute_length_factor(chromosome, rec_map, demo, reference_genome)
        filter_array = np.full(len(maf), False)
        ukbb_freq_hist = './files/ukbb-freq/chr' + str(chromosome) + '.hist'
        with open(ukbb_freq_hist) as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.split()
                min_maf = float(line_split[0])
                max_maf = float(line_split[1])
                n_variants = int(int(line_split[2]) * length_factor)
                relevant = (maf > min_maf) & (maf <= max_maf)
                relevant_index = np.arange(len(maf))[relevant]
                if len(relevant_index) > 0:
                    keep_index = np.random.choice(relevant_index,
                                                  size=min(n_variants, len(relevant_index)),
                                                  replace=False)
                    filter_array[keep_index] = True
        # np.random.seed()
        return filter_array

    @staticmethod
    def remove_duplicates(phys_pos):
        """Remove duplicates by incrementing duplicate positions as much as necessary."""
        for index in range(1, len(phys_pos)):
            increment = 1
            cur_index = index
            duplicate = False
            while cur_index < len(phys_pos) and phys_pos[cur_index] == phys_pos[index - 1]:
                phys_pos[cur_index] = phys_pos[index] + increment
                increment += 1
                cur_index += 1
                duplicate = True
            if duplicate:
                phys_pos = np.sort(phys_pos)  # avoid issue with pos such as [..., 6, 6, 6, 7, 8, 9, ...]
        assert len(phys_pos) == len(np.unique(phys_pos)), 'Duplicate positions found, will generate errors with ASMC.'
        return phys_pos

    @staticmethod
    def get_genetic_positions(phys_pos, genetic_map, config):
        """Return the genetic positions."""

        if hasattr(config, 'rec_rate') and hasattr(config, 'length'):
            gen_pos = phys_pos * config.rec_rate / 1e-2  # divide by 1e-2 for cM
        else:
            gen_pos = []
            cur_g = 0
            for index_pos, bp in enumerate(phys_pos):

                while bp > genetic_map[cur_g][0] and cur_g < len(genetic_map) - 1:
                    cur_g += 1

                if bp >= genetic_map[cur_g][0]:
                    # we found the exact marker, or reached the end of the map
                    cm = genetic_map[cur_g][2]
                elif cur_g == 0:
                    # if we haven't hit the map yet, store first map entry
                    cm = genetic_map[cur_g][2]
                else:
                    # interpolate from previous marker
                    cm = genetic_map[cur_g - 1][2] \
                         + (bp - genetic_map[cur_g - 1][0]) * (genetic_map[cur_g][2] - genetic_map[cur_g - 1][2]) \
                         / (genetic_map[cur_g][0] - genetic_map[cur_g - 1][0])

                gen_pos.append(cm)

        return np.asarray(gen_pos)

    @staticmethod
    def create_dataset_name(config, mode, chromosome, seed):
        """Create path to dataset."""

        populations = ["CEU.Terhorst", "ACB", "CEU", "ESN", "GWD", "KHV", "PEL", "STU", "ASW", "CHB", "FIN", "IBS",
                       "LWK", "PJL", "TSI", "BEB", "CHS", "GBR", "ITU", "MSL", "PUR", "YRI", "CDX",
                       "CLM", "GIH", "JPT", "MXL"]
        if "constant" in config.demography:
            demo = 'constant' + '.Ne.' + str(config.Ne)
        elif config.demography in populations:
            demo = str(config.demography)
        else:
            raise ValueError('Demographic model is unknown.')

        recombination = ''
        if 'rec_rate' in config:
            recombination = '.rec_const.' + str(config.rec_rate) \
                            + '.length.' + str(config.length)

        if mode == 'train':
            dataset_name = 'CHR.' + str(chromosome) \
                           + '.S.' + str(config.sample_size_train) \
                           + '.DEMO.' + demo \
                           + recombination \
                           + '.seed.' + str(seed)
        elif mode == 'val':
            dataset_name = 'CHR.' + str(chromosome) \
                           + '.S.' + str(config.sample_size_val) \
                           + '.DEMO.' + demo \
                           + recombination \
                           + '.seed.' + str(seed)
        elif mode == 'run':
            dataset_name = 'CHR.' + str(chromosome) \
                           + '.S.' + str(config.sample_size_run) \
                           + '.DEMO.' + demo \
                           + recombination \
                           + '.seed.' + str(seed)

        return dataset_name

    @staticmethod
    def count_num_features(config):
        """Count number of features."""

        num_features = 2  # haplotypes for a pair
        if config.feature_maf:
            num_features += 1
        if config.feature_cm:
            num_features += 1
        if config.feature_bp:
            num_features += 1
        if config.feature_ibs:
            num_features += 1
        # if config.feature_demo_const:
        #    num_features += 1
        # if self.data_type == 'impute':
        #    num_features += 1
        return num_features

    @staticmethod
    def crop_rec_map(config, chromosome, dataset_name, genetic_map):
        file_name = os.path.join(config.session_name, dataset_name + '.temp.map.gz')
        if genetic_map is not None:
            f = gzip.open(file_name, "wt")
            f.write('chr' + str(chromosome) + '\t'
                    + 'position' + '\t'
                    + 'COMBINED_rate(cM/Mb)' + '\t'
                    + 'Genetic_Map(cM)' + '\n')
            for pos in genetic_map:
                f.write('chr' + str(chromosome) + '\t'
                        + str(pos[0]) + '\t'
                        + str(pos[1]) + '\t'
                        + str(pos[2]) + '\n')
            f.close()
        return file_name

    def crop_genetic_map(self, genetic_map, mode):
        if genetic_map is not None:
        # genetic_map is a list of (phys_pos, rec, gen_pos) tuples
            if mode == 'train':
                # get random window of 20 cM
                window_size = 30
                gen_pos = [pos[2] for pos in genetic_map]
                max_gen_pos = gen_pos[-1] - window_size  # 30 cM windows
                max_gen_pos = max(max_gen_pos, 1)  # needs to be strictly positive
                # get max index to sample from
                max_index = 0
                while max_index < len(gen_pos) and gen_pos[max_index] < max_gen_pos:
                    max_index += 1
                # compute probability weights for sampling
                phys_pos = np.asarray([pos[0] for pos in genetic_map[:max_index]])
                prev_phys_pos = np.zeros_like(phys_pos)
                prev_phys_pos[1:] = phys_pos[:-1]
                distances = phys_pos - prev_phys_pos
                weights = distances / distances.sum()
                # get start index to sample from
                start_index = np.random.choice(range(0, max_index), p=weights)
                # get end index to sample from
                end_index = start_index
                while end_index < len(gen_pos) and gen_pos[end_index] - gen_pos[start_index] < window_size:
                    end_index += 1

            elif mode == 'val':
                # get first 20Mbp
                window_size = 30 * 1e6
                phys_pos = [pos[0] for pos in genetic_map]
                start_index = 0
                end_index = 0
                while phys_pos[end_index] < window_size:
                    end_index += 1

            elif mode == 'run':
                # get first self.config.chr_length Mbp
                window_size = self.config.chr_length * 1e6
                phys_pos = [pos[0] for pos in genetic_map]
                start_index = 0
                end_index = 0
                while phys_pos[end_index] < window_size:
                    end_index += 1

            # crop genetic map and update phys and gen positions
            cropped_genetic_map = copy.deepcopy(genetic_map[start_index: end_index + 1])
            if start_index > 0:
                start_phys_pos = cropped_genetic_map[0][0]
                start_gen_pos = cropped_genetic_map[0][2]
                for i in range(len(cropped_genetic_map)):
                    cropped_genetic_map[i][0] = cropped_genetic_map[i][0] - start_phys_pos
                    cropped_genetic_map[i][2] = cropped_genetic_map[i][2] - start_gen_pos
            # last rate should be zero
            cropped_genetic_map[-1][1] = 0
        else:
            cropped_genetic_map = None

        return cropped_genetic_map

    @staticmethod
    def read_genetic_map(rec_map):
        """Read genetic map."""

        # print('Computing genetic positions...')
        if rec_map != '':
            genetic_map = []  # list of (phys_pos, rec, gen_pos) tuples
            with gzip.open(rec_map, 'rt') as file_map:
                line = file_map.readline()
                cur_g = 0
                while line:
                    line_split = line.split()
                    if line_split[1] == 'position' or line[1] == '' or line_split[1] == 'Position(bp)':
                        line = file_map.readline()
                        continue
                    genetic_map.append([int(line_split[1]), float(line_split[2]), float(line_split[3])])
                    if cur_g > 0 and (genetic_map[cur_g][0] < genetic_map[cur_g - 1][0] or genetic_map[cur_g][2] <
                                    genetic_map[cur_g - 1][2]):
                        raise ValueError('Genetic map not in sorted order at line {}'.format(line))
                    cur_g += 1
                    line = file_map.readline()
        else:
            genetic_map = None

        return genetic_map