"""
Code adapted from Anthony Hu.
https://github.com/anthonyhu/ml-research
"""

import os
import shutil
import sys
from abc import ABCMeta, abstractmethod

import yaml
import torch
import torch.nn.functional as F
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time, sleep
import gzip
import datetime
import socket
import json
import subprocess
import tsinfer
import tsdate
import cyvcf2
import msprime

from utils import Config, Logger, print_model_spec, track, save_numpy
from scipy.stats import gmean
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.ensemble import AdaBoostClassifier


class Dating:
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
        self.singletons = None
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

    @staticmethod
    def compute_l1_loss(prediction, ground_truth):
        return np.nanmean(np.abs(prediction - ground_truth))

    @staticmethod
    def compute_l2_loss(prediction, ground_truth):
        return np.sqrt(np.nanmean((prediction - ground_truth) ** 2))

    @staticmethod
    def write_map(simulation, first_column=False):
        path = simulation['path_dataset']
        phys_pos = simulation['phys_pos']
        gen_pos = simulation['gen_pos']
        chromosome = simulation['chromosome']
        rate = ((gen_pos[1:] - gen_pos[:-1]) / (phys_pos[1:] - phys_pos[:-1])) * 10**6

        with open(path + '.map', 'w') as out_file:

            if first_column:
                out_file.write('Chromosome\t' +
                               'Position(bp)\t' +
                               'Rate(cM/Mb)\t' +
                               'Map(cM)\n')
            else:
                out_file.write('Position(bp)\t' +
                               'Rate(cM/Mb)\t' +
                               'Map(cM)\n')

            for i in range(0, len(phys_pos) - 1):
                if first_column:
                    out_file.write('\t'.join(['chr' + str(chromosome),
                                              str(phys_pos[i]),
                                              str(rate[i]),
                                              str(gen_pos[i])]) + '\n')
                else:
                    out_file.write('\t'.join([str(phys_pos[i]),
                                              str(rate[i]),
                                              str(gen_pos[i])]) + '\n')

            # add last row with zero rate
            if first_column:
                out_file.write('\t'.join(['chr' + str(chromosome),
                                          str(phys_pos[-1]),
                                          "0",
                                          str(gen_pos[-1])]) + '\n')
            else:
                out_file.write('\t'.join([str(phys_pos[-1]),
                                          "0",
                                          str(gen_pos[-1])]) + '\n')

        return

    @staticmethod
    def write_hap(simulation):
        path = simulation['path_dataset']
        phys_pos = simulation['phys_pos']
        genotype_matrix = simulation['genotype_matrix']
        chromosome = simulation['chromosome']
        print("Writing " + path + ".haps ...")
        with open(path + '.haps', 'w') as out_file:
            for index, pos in enumerate(phys_pos):
                row_list = [str(chromosome), 'SNP' + str(index), str(int(pos)), 'A', 'G']
                row_list += [str(int(round(entry))) for entry in genotype_matrix[:, index]]
                out_file.write(' '.join(row_list))
                out_file.write('\n')
        return

    @staticmethod
    def write_samples(simulation, num_diploids):
        path = simulation['path_dataset']
        print("Writing " + path + ".samples ...")
        with open(path + '.samples', 'w') as out_file:
            out_file.write('\t'.join(["ID_1", "ID_2", "missing"]) + '\n')
            out_file.write('\t'.join(["0", "0", "0"]) + '\n')
            for i in range(1, num_diploids + 1):
                out_file.write('\t'.join(["sample_" + str(i), "sample_" + str(i), "0"]) + '\n')
        return

    def compute_and_save_metrics(self, prediction, label, name):
        is_prediction_nan = np.isnan(prediction)
        non_nan_prediction = prediction[np.logical_not(is_prediction_nan)]
        non_nan_label = label[np.logical_not(is_prediction_nan)]
        rmse = mean_squared_error(non_nan_label, non_nan_prediction, squared=False)
        mean_absolute_err = mean_absolute_error(non_nan_label, non_nan_prediction)
        median_absolute_err = median_absolute_error(non_nan_label, non_nan_prediction)
        r2 = stats.pearsonr(non_nan_label, non_nan_prediction)[0] ** 2
        # r2 = r2_score(non_nan_label, non_nan_prediction)  getting weird results (negative r2 score)
        self.save_metrics_json(name, rmse, mean_absolute_err, median_absolute_err, r2)
        return rmse, mean_absolute_err, median_absolute_err, r2

    @track
    def run(self):
        print('Running session..')
        num_variants = self.run_dataset.simulations[0]['num_variants']
        self.config.num_variants = min(self.config.num_variants, num_variants)

        mutations_age = self.run_dataset.simulations[0]['mutations_age']

        self.singletons = np.sum(self.run_dataset.simulations[0]['genotype_matrix'], axis=0) == 1

        if self.config.remove_singletons:
            # insert NaN for singleton variants
            mutations_age[self.singletons] = float('nan')

        ground_truth = mutations_age[:self.config.num_variants]
        save_numpy(self.session_name, ground_truth, 'ground_truth_ages')
        print('Total number of variants to date:', self.config.num_variants)
        cm_distance = self.run_dataset.simulations[0]['gen_pos'][self.config.num_variants - 1]
        bp_distance = self.run_dataset.simulations[0]['phys_pos'][self.config.num_variants - 1]
        print('This corresponds to', cm_distance, 'cM and', bp_distance, 'bp.')

        if self.config.run_CoalNN:
            deep_coalescent_min_ages, deep_coalescent_max_ages, deep_coalescent_ages = self.run_CoalNN()
            deep_coalescent_min_ages = deep_coalescent_min_ages[:self.config.num_variants]
            deep_coalescent_max_ages = deep_coalescent_max_ages[:self.config.num_variants]
            deep_coalescent_ages = deep_coalescent_ages[:self.config.num_variants]
            save_numpy(self.session_name, deep_coalescent_min_ages, 'coalNN_min_ages')
            save_numpy(self.session_name, deep_coalescent_max_ages, 'coalNN_max_ages')
            save_numpy(self.session_name, deep_coalescent_ages, 'coalNN_ages')
            deep_coalescent_ages_nan_count = np.count_nonzero(~np.isnan(deep_coalescent_ages))
            print('Number of dated variants in deep coalescent prediction:', deep_coalescent_ages_nan_count)
            rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(deep_coalescent_ages, ground_truth, 'coalNN')
            print('RMSE deep coalescent: {:.3f}'.format(rmse))
            print('Mean absolute error deep coalescent: {:.3f}'.format(mean_abs))
            print('Median absolute error deep coalescent: {:.3f}'.format(med_abs))
            print('R2 score deep coalescent: {:.3f}\n'.format(r2))
            non_nan_deep_coalescent = np.logical_not(np.isnan(deep_coalescent_ages))

        if self.config.geva:
            geva_ages = self.run_geva()
            geva_ages = geva_ages[:self.config.num_variants]
            save_numpy(self.session_name, geva_ages, 'geva_ages')
            geva_ages_nan_count = np.count_nonzero(~np.isnan(geva_ages))
            print('Number of dated variants in geva prediction:', geva_ages_nan_count)
            rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(geva_ages, ground_truth, 'geva')
            print('RMSE geva: {:.3f}'.format(rmse))
            print('Mean absolute error geva: {:.3f}'.format(mean_abs))
            print('Median absolute error geva: {:.3f}'.format(med_abs))
            print('R2 score geva: {:.3f}\n'.format(r2))
            non_nan_geva = np.logical_not(np.isnan(geva_ages))

        if self.config.tsdate:
            tsdate_ages = self.run_tsdate()
            # tsdate_min_ages = tsdate_min_ages[:self.config.num_variants]
            # tsdate_max_ages = tsdate_max_ages[:self.config.num_variants]
            tsdate_ages = tsdate_ages[:self.config.num_variants]
            # save_numpy(self.session_name, tsdate_min_ages, 'tsdate_min_ages')
            # save_numpy(self.session_name, tsdate_max_ages, 'tsdate_max_ages')
            save_numpy(self.session_name, tsdate_ages, 'tsdate_ages')
            tsdate_ages_nan_count = np.count_nonzero(~np.isnan(tsdate_ages))
            print('Number of dated variants in tsdate prediction:', tsdate_ages_nan_count)
            rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(tsdate_ages, ground_truth, 'tsdate')
            print('RMSE tsdate: {:.3f}'.format(rmse))
            print('Mean absolute error tsdate: {:.3f}'.format(mean_abs))
            print('Median absolute error tsdate: {:.3f}'.format(med_abs))
            print('R2 score tsdate: {:.3f}\n'.format(r2))
            non_nan_tsdate = np.logical_not(np.isnan(tsdate_ages))

        if self.config.relate:
            relate_min_ages, relate_max_ages, relate_ages = self.run_relate()
            relate_min_ages = relate_min_ages[:self.config.num_variants]
            relate_max_ages = relate_max_ages[:self.config.num_variants]
            relate_ages = relate_ages[:self.config.num_variants]
            save_numpy(self.session_name, relate_min_ages, 'relate_min_ages')
            save_numpy(self.session_name, relate_max_ages, 'relate_max_ages')
            save_numpy(self.session_name, relate_ages, 'relate_ages')
            relate_ages_nan_count = np.count_nonzero(~np.isnan(relate_ages))
            print('Number of dated variants in relate prediction:', relate_ages_nan_count)
            rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(relate_ages, ground_truth, 'relate')
            print('RMSE relate: {:.3f}'.format(rmse))
            print('Mean absolute error relate: {:.3f}'.format(mean_abs))
            print('Median absolute error relate: {:.3f}'.format(med_abs))
            print('R2 score relate: {:.3f}\n'.format(r2))
            non_nan_relate = np.logical_not(np.isnan(relate_ages))

        do_compare = (self.config.run_CoalNN and self.config.geva) or \
                     (self.config.run_CoalNN and self.config.tsdate) or \
                     (self.config.run_CoalNN and self.config.relate)

        if do_compare:

            overlap_variants = None
            if self.config.run_CoalNN and self.config.geva:
                overlap_variants = np.logical_and(non_nan_deep_coalescent, non_nan_geva)
            if self.config.run_CoalNN and self.config.tsdate:
                if overlap_variants is None:
                    overlap_variants = np.logical_and(non_nan_deep_coalescent, non_nan_tsdate)
                else:
                    overlap_variants = np.logical_and(overlap_variants, non_nan_tsdate)
            if self.config.run_CoalNN and self.config.relate:
                if overlap_variants is None:
                    overlap_variants = np.logical_and(non_nan_deep_coalescent, non_nan_relate)
                else:
                    overlap_variants = np.logical_and(overlap_variants, non_nan_relate)
            save_numpy(self.session_name, overlap_variants, 'overlap_variants_ages')

            print('-' * 100)
            print('Number of overlapping dated variants', np.count_nonzero(overlap_variants))
            ground_truth = ground_truth[overlap_variants]

            deep_coalescent_ages = deep_coalescent_ages[overlap_variants]
            rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(deep_coalescent_ages, ground_truth, 'overlap_coalNN')
            print('\nRMSE deep coalescent: {:.3f}'.format(rmse))
            print('Mean absolute error deep coalescent: {:.3f}'.format(mean_abs))
            print('Median absolute error deep coalescent: {:.3f}'.format(med_abs))
            print('R2 score deep coalescent: {:.3f}\n'.format(r2))

            if self.config.geva:
                geva_ages = geva_ages[overlap_variants]
                rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(geva_ages, ground_truth, 'overlap_geva')
                print('RMSE geva: {:.3f}'.format(rmse))
                print('Mean absolute error geva: {:.3f}'.format(mean_abs))
                print('Median absolute error geva: {:.3f}'.format(med_abs))
                print('R2 score geva: {:.3f}\n'.format(r2))

            if self.config.tsdate:
                tsdate_ages = tsdate_ages[overlap_variants]
                rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(tsdate_ages, ground_truth, 'overlap_tsdate')
                print('RMSE tsdate: {:.3f}'.format(rmse))
                print('Mean absolute error tsdate: {:.3f}'.format(mean_abs))
                print('Median absolute error tsdate: {:.3f}'.format(med_abs))
                print('R2 score tsdate: {:.3f}\n'.format(r2))

            if self.config.relate:
                relate_ages = relate_ages[overlap_variants]
                rmse, mean_abs, med_abs, r2 = self.compute_and_save_metrics(relate_ages, ground_truth, 'overlap_relate')
                print('RMSE relate: {:.3f}'.format(rmse))
                print('Mean absolute error relate: {:.3f}'.format(mean_abs))
                print('Median absolute error relate: {:.3f}'.format(med_abs))
                print('R2 score relate: {:.3f}\n'.format(r2))

    def run_relate(self):
        print('Starting relate session..')
        print('Will decode', len(self.random_pairs), 'pairs...')

        simulation = self.run_dataset.simulations[0]
        path = simulation['path_dataset']
        dataset_name = simulation['dataset_name']
        num_variants = simulation['num_variants']

        # Prepare files
        Dating.write_hap(simulation)
        Dating.write_map(simulation)
        Dating.write_samples(simulation, self.run_dataset.num_diploids)

        # Run Relate
        coal_file = None
        if self.config.demography == 'CEU.Terhorst':
            coal_file = "./files/coal/CEU.Terhorst.coal"
        elif self.config.demography == 'CEU':
            coal_file = "./files/coal/CEU.popsizes_1kg.generations.mu1.65E-8.haploids.coal"
        if coal_file is not None:
            subprocess.run(["../relate_v1.1.6_x86_64_dynamic/bin/Relate",
                            "--mode", "All",
                            "-m", "1.65e-8",
                            "--coal", coal_file,
                            "--haps", path + ".haps",
                            "--sample", path + ".samples",
                            "--map", path + ".map",
                            "-o", dataset_name + ".relate"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL
                           )
        else:
            subprocess.run(["../relate_v1.1.6_x86_64_dynamic/bin/Relate",
                            "--mode", "All",
                            "-m", str(self.config.muration_rate),
                            "-N", str(2 * self.config.Ne),
                            "--haps", path + ".haps",
                            "--sample", path + ".samples",
                            "--map", path + ".map",
                            "-o", dataset_name + ".relate"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL
                           )

        # relate requires output files to be in working directory, now let's move them
        relate_path = os.path.join(self.session_name, dataset_name + ".relate.mut")
        shutil.move(dataset_name + ".relate.mut", relate_path)
        relate_path = os.path.join(self.session_name, dataset_name + ".relate.anc")
        shutil.move(dataset_name + ".relate.anc", relate_path)

        def read_relate_age(path):
            ages_begin = float('nan') * np.ones(num_variants)
            ages_end = float('nan') * np.ones(num_variants)
            with open(path + '.relate.mut', "r") as relate_file:
                lines = relate_file.readlines()
                for i, line in enumerate(lines[1:], 1):
                    if i > self.config.num_variants:
                        break
                    line_split = line.split(';')
                    index_snp = int(line_split[0])
                    age_begin = float(line_split[8])
                    age_end = float(line_split[9])
                    ages_begin[index_snp] = age_begin
                    ages_end[index_snp] = age_end
            return ages_begin, ages_end

        min_mutations_age, max_mutations_age = read_relate_age(path)
        variants_ages = np.mean([min_mutations_age, max_mutations_age], axis=0)
        # variants_ages = gmean([min_mutations_age, max_mutations_age], axis=0)

        if self.config.remove_singletons:
            # insert NaN for singleton variants
            variants_ages[self.singletons] = float('nan')
            min_mutations_age[self.singletons] = float('nan')
            max_mutations_age[self.singletons] = float('nan')

        return min_mutations_age, max_mutations_age, variants_ages

    @track
    def run_tsdate(self):
        print('Starting tsdate session..')
        print('Will decode', len(self.random_pairs), 'pairs...')

        def add_diploid_sites(vcf, samples, phys_pos):
            """
            Read the sites in the vcf and add them to the samples object, reordering the
            alleles to put the ancestral allele first, if it is available.
            """
            for i, variant in enumerate(vcf):  # Loop over variants, each assumed at a unique site
                pos = phys_pos[i]
                if any([not phased for _, _, phased in variant.genotypes]):
                    raise ValueError("Unphased genotypes for variant at position", pos)
                alleles = [variant.REF] + variant.ALT
                ancestral = variant.INFO.get("AA", variant.REF)
                # Ancestral state must be first in the allele list.
                ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})
                allele_index = {
                    old_index: ordered_alleles.index(allele)
                    for old_index, allele in enumerate(alleles)
                }
                # Map original allele indexes to their indexes in the new alleles list.
                genotypes = [
                    allele_index[old_index]
                    for row in variant.genotypes
                    for old_index in row[0:2]
                ]
                samples.add_site(pos, genotypes=genotypes, alleles=alleles)
            samples.finalise()

        def chromosome_length(vcf):
            assert len(vcf.seqlens) == 1
            return vcf.seqlens[0]

        def get_mutations_age(tree_sequence, phys_pos):
            min_mutations_age = []
            max_mutations_age = []

            for tree in tree_sequence.trees():
                for mutation in tree.mutations():
                    node = mutation.node
                    parent = tree.get_parent(node)
                    min_time = tree.get_time(node)
                    if parent == -1:
                        max_time = float('nan')
                    else:
                        max_time = tree.get_time(parent)
                    min_mutations_age.append(min_time)
                    max_mutations_age.append(max_time)
            assert len(min_mutations_age) == len(phys_pos), 'Error while computing min mutations age.'
            assert len(max_mutations_age) == len(phys_pos), 'Error while computing max mutations age.'
            return np.asarray(min_mutations_age), np.asarray(max_mutations_age)

        simulation = self.run_dataset.simulations[0]
        phys_pos = simulation['phys_pos']
        chromosome = simulation['chromosome']
        path = simulation['path_dataset']

        # Prepare recombination map
        Dating.write_map(simulation, first_column=True)
        recombination_map = msprime.RateMap.read_hapmap(path + ".map")
        # recombination_map = None

        # write vcf
        with gzip.open(path + ".vcf.gz", "wt") as vcf_file:
            simulation['tree_sequence'].write_vcf(vcf_file)

        # get rid of unecessary variants
        # output_file = gzip.open(path + ".tsdate.vcf.gz", "w")
        # p1 = subprocess.Popen("zcat " + path + ".vcf.gz", shell=True, stdout=subprocess.PIPE)
        # p2 = subprocess.Popen("head -n " + str(self.config.num_variants + 50), shell=True, stdin=p1.stdout, stdout=output_file)
        # output_file.close()

        # infer tree with tsinfer
        vcf = cyvcf2.VCF(path + ".vcf.gz")
        with tsinfer.SampleData(path=path + ".samples.gz", sequence_length=chromosome_length(vcf)) as samples:
            add_diploid_sites(vcf, samples, phys_pos)
            inferred_ts = tsinfer.infer(samples, recombination_rate=recombination_map).simplify()
        # samples = tsinfer.SampleData(path=path + ".samples.gz", sequence_length=chromosome_length(vcf))
        # samples = add_diploid_sites(vcf, samples, phys_pos)
        # inferred_ts = tsinfer.infer(samples)

        # infer ages with tsdate
        dated_ts = tsdate.date(inferred_ts, Ne=self.config.Ne, mutation_rate=self.config.muration_rate)
        variants_ages = tsdate.sites_time_from_ts(dated_ts, node_selection='arithmetic')
        # variants_ages = tsdate.sites_time_from_ts(dated_ts, node_selection='geometric')

        if self.config.remove_singletons:
            # insert NaN for singleton variants
            variants_ages[self.singletons] = float('nan')

        return variants_ages

    @track
    def run_geva(self):

        def read_geva_age(path):
            if os.path.exists(path + '.sites.txt'):
                with open(path + '.sites.txt', "r") as geva_file:
                    lines = geva_file.readlines()
                    if len(lines) == 7:
                        line = lines[-1].split()  # joint clock after heuristic filtering of pairs
                        age = float(line[-2])  # mode of the composite posterior distribution
                    else:
                        age = float('nan')
                        # raise ValueError('Found more or less than 7 lines in ' + path + '.sites.txt')
                os.remove(path + '.sites.txt')
                os.remove(path + '.pairs.txt')
                os.remove(path + '.log')
                os.remove(path + '.err')
            else:
                age = float('nan')
            return age

        print('Starting GEVA session..')
        print('Will decode', len(self.random_pairs), 'pairs...')

        simulation = self.run_dataset.simulations[0]
        phys_pos = simulation['phys_pos']
        chromosome = simulation['chromosome']
        path = simulation['path_dataset']

        # write vcf
        with gzip.open(path + ".vcf.gz", "wt") as vcf_file:
            simulation['tree_sequence'].write_vcf(vcf_file)

        # write map
        Dating.write_map(simulation)

        # preparing files
        subprocess.run(["../geva/geva_v1beta",
                        "--vcf", path + ".vcf.gz",
                        "--map", path + ".map",
                        "--out", path + ".geva"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL
                       )
        geva_path = os.path.join(self.session_name, 'geva')
        os.makedirs(geva_path)

        # batch version, which does not seem to work... (see issue on github)
        # write batch file
        # with open(path + ".geva.batch", "wt") as batch_file:
        #    for i in range(0, len(phys_pos)):
        #        batch_file.write(str(phys_pos[i]) + '\n')

        # subprocess.run(["../geva/geva_v1beta",
        #                "-i", path + ".geva.bin",
        #                "-o", path + ".geva.run",
        #                "--positions", path + ".geva.batch",
        #                "--Ne", "10000",
        #                "--mut", "1e-8",
        #                "--hmm", "../geva/hmm/hmm_initial_probs.txt", "../geva/hmm/hmm_emission_probs.txt"])
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
        # )

        num_variants = self.run_dataset.simulations[0]['num_variants']
        variants_ages = float('nan') * np.ones(num_variants)
        bin_size = self.x_dim[-1]

        for i, pos in tqdm(enumerate(phys_pos), total=self.config.num_variants):

            if i == self.config.num_variants:
                break

            geva_path_file = geva_path + "/run." + str(pos)
            subprocess.run(["../geva/geva_v1beta",
                            "-i", path + ".geva.bin",
                            "-o", geva_path_file,
                            "--position", str(pos),
                            "--Ne", "10000",
                            "--mut", "1e-8",
                            "--maxConcordant", str(self.run_dataset.num_pairs),
                            "--maxDiscordant", str(self.run_dataset.num_pairs),
                            "--hmm", "../geva/hmm/hmm_initial_probs.txt", "../geva/hmm/hmm_emission_probs.txt"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            variants_ages[i] = read_geva_age(geva_path_file)

        if self.config.remove_singletons:
            # insert NaN for singleton variants
            variants_ages[self.singletons] = float('nan')

        return variants_ages

    def compute_concordant_discordant_prediction(self, batch, prediction):
        if self.config.data_type == 'impute':
            hap_1_no_mut = batch['input'][:, 0, self.context_size:-self.context_size] == 0
            hap_2_no_mut = batch['input'][:, 1, self.context_size:-self.context_size] == 0
            hap_1_mut = batch['input'][:, 0, self.context_size:-self.context_size] == 1
            hap_2_mut = batch['input'][:, 1, self.context_size:-self.context_size] == 1
            concordant_pairs = torch.logical_and(hap_1_mut, hap_2_mut)
            discordant_pairs = torch.logical_and(hap_1_no_mut, hap_2_mut) | \
                               torch.logical_and(hap_1_mut, hap_2_no_mut)
        else:
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
            best_threshold = decision_tree.threshold[decision_tree.feature == 0][0]
            concordant_ages[concordant_ages >= best_threshold] = float('nan')
            discordant_ages[discordant_ages < best_threshold] = float('nan')
        elif len(discordant_ages_non_nan) > 0 and len(concordant_ages_non_nan) == 0:
            # singleton
            concordant_ages[:] = 0
        elif len(discordant_ages_non_nan) == 0 and len(concordant_ages_non_nan) > 0:
            # all samples have the mutation (which should never happen..?)
            discordant_ages[:] = np.nanmax(concordant_ages)
        else:
            # any other scenario (which should never happen..?)
            concordant_ages[:] = float('nan')
            discordant_ages[:] = float('nan')

    def filter_outliers(self, concordant_ages, discordant_ages, bin_size, start_pos):

        clf = AdaBoostClassifier(n_estimators=1, random_state=0)
        # clf = SVC(kernel='linear')

        for variant in range(bin_size):

            if start_pos + variant > self.config.num_variants:
                break

            self.filter_outliers_axis(concordant_ages[:, variant], discordant_ages[:, variant], clf)

        return concordant_ages, discordant_ages

    @track
    def run_CoalNN(self):
        print('Starting CoalNN session..')
        print('Will decode', len(self.random_pairs), 'pairs...')
        start_time = time()
        self.model.eval()

        with torch.no_grad():

            num_variants = self.run_dataset.simulations[0]['num_variants']
            concordant_ages = float('nan') * np.ones(num_variants)
            discordant_ages = float('nan') * np.ones(num_variants)

            for pos_iteration, start_pos in tqdm(enumerate(range(0,
                                                                 num_variants + 2 * self.context_size - self.input_size,
                                                                 self.focus_input_size)),
                                                 total=num_variants // self.focus_input_size):

                start_label = pos_iteration * self.focus_input_size
                if start_label > self.config.num_variants:
                    break

                concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size), device=self.device)
                discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size), device=self.device)
                # concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size))
                # discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size))

                prev_batch_size = self.config.batch_size

                for iteration, batch in tqdm(enumerate(self.run_dataloader), total=len(self.run_dataloader)):
                    sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                    prediction = self.run_prediction(sub_batch)
                    prediction = prediction['output']

                    concordant_prediction, discordant_prediction = \
                        self.compute_concordant_discordant_prediction(sub_batch, prediction)

                    cur_batch_size = prediction.shape[0]
                    concordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] = \
                        concordant_prediction
                    discordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size, :] = \
                        discordant_prediction
                    prev_batch_size = cur_batch_size

                concordant_ages_bin = concordant_ages_bin.cpu().numpy()
                discordant_ages_bin = discordant_ages_bin.cpu().numpy()
                # concordant_ages_bin = concordant_ages_bin.numpy()
                # discordant_ages_bin = discordant_ages_bin.numpy()

                if self.config.filter_outliers:
                    concordant_ages_bin, discordant_ages_bin = \
                        self.filter_outliers(concordant_ages_bin, discordant_ages_bin, self.focus_input_size, start_label)

                concordant_ages[start_label: start_label + self.focus_input_size] = np.nanmax(concordant_ages_bin, axis=0)
                discordant_ages[start_label: start_label + self.focus_input_size] = np.nanmin(discordant_ages_bin, axis=0)

            if start_label < self.config.num_variants:
                # run last bin if necessary
                if num_variants % self.focus_input_size != 0:
                    start_pos = num_variants + 2 * self.context_size - self.input_size
                    concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size), device=self.device)
                    discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size), device=self.device)
                    # concordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size))
                    # discordant_ages_bin = torch.zeros((self.run_dataset.num_pairs, self.focus_input_size))

                    prev_batch_size = self.config.batch_size

                    for iteration, batch in tqdm(enumerate(self.run_dataloader), total=len(self.run_dataloader)):
                        sub_batch = {'input': batch['input'][:, :, start_pos: start_pos + self.input_size]}
                        prediction = self.run_prediction(sub_batch)
                        prediction = prediction['output']

                        concordant_prediction, discordant_prediction = \
                            self.compute_concordant_discordant_prediction(sub_batch, prediction)

                        cur_batch_size = prediction.shape[0]
                        concordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size,
                        :] = \
                            concordant_prediction
                        discordant_ages_bin[iteration * prev_batch_size: iteration * prev_batch_size + cur_batch_size,
                        :] = \
                            discordant_prediction
                        prev_batch_size = cur_batch_size

                    concordant_ages_bin = concordant_ages_bin.cpu().numpy()
                    discordant_ages_bin = discordant_ages_bin.cpu().numpy()
                    # concordant_ages_bin = concordant_ages_bin.numpy()
                    # discordant_ages_bin = discordant_ages_bin.numpy()

                    start_label = num_variants - self.focus_input_size
                    concordant_ages_bin, discordant_ages_bin = \
                        self.filter_outliers(concordant_ages_bin, discordant_ages_bin, self.focus_input_size, start_label)

                    concordant_ages[start_label: start_label + self.focus_input_size] = np.nanmax(concordant_ages_bin, axis=0)
                    discordant_ages[start_label: start_label + self.focus_input_size] = np.nanmin(discordant_ages_bin, axis=0)

            # artihmetic_mean = np.mean([concordant_ages, discordant_ages], axis=0)
            # geometric_mean = gmean([concordant_ages, discordant_ages], axis=0)
            # variants_ages = np.where(concordant_ages < 10**3, concordant_ages, artihmetic_mean)

            if self.config.dating_mode == 'arithmetic':
                variants_ages = np.mean([concordant_ages, discordant_ages], axis=0)
            elif self.config.dating_mode == 'geometric':
                variants_ages = gmean([concordant_ages, discordant_ages], axis=0)
            else:
                raise ValueError('Dating mode for CoalNN is unknown.')

            if self.config.remove_singletons:
                # insert NaN for singleton variants
                variants_ages[self.singletons] = float('nan')
                concordant_ages[self.singletons] = float('nan')
                discordant_ages[self.singletons] = float('nan')

            time_duration = time() - start_time
            print('CoalNN decoding done in : {:.0f}ms'.format(1000 * time_duration))

            return concordant_ages, discordant_ages, variants_ages

    @staticmethod
    def get_sample_const_piecewise(sample):
        prediction = sample[0]
        breakpoints = sample[1]
        threshold = sample[2]
        breakpoints = breakpoints[1, :]  # class 1
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

    def run_prediction(self, batch):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        output['breakpoints'] = F.softmax(output['breakpoints'], dim=1)
        if self.config.log_tmrca:
            output['output'] = torch.exp(output['output'])
        if self.config.constant_piecewise_output:
            output['output'] = self.get_const_piecewise(copy.deepcopy(output), self.config.const_threshold)
        return output

    def save_metrics_json(self, name, rmse, mean_abs_err, med_abs_err, r2):
        filename = os.path.join(self.session_name, name + '_dating_metrics.json')
        output = {'rmse': float("{:.3f}".format(rmse)),
                  'mean_abs_err': float("{:.3f}".format(mean_abs_err)),
                  'med_abs_err': float("{:.3f}".format(med_abs_err)),
                  'r2': float("{:.3f}".format(r2))}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def save_segments_json(self, name, ground_truth, prediction):
        filename = os.path.join(self.session_name, name + '_metrics.json')
        output = {'ground_truth': ground_truth, 'prediction': prediction}
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

        if self.options.const_threshold:
            config_run['constant_piecewise_output'] = True
            config_run['const_threshold'] = float(self.options.const_threshold)

        if self.options.downsample_size:
            config_run['downsample_size'] = int(self.options.downsample_size)

        self.config.update_config(config_run)
        self.restore_session_name = self.config.session_name

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
        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(config_run, f)
