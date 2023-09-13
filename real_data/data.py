from torch.utils.data import Dataset
import numpy as np
import gzip
import torch
from time import time


class RealDataset(Dataset):
    """
    Real dataset, currently only supports WGS data.
    """

    def __init__(self, config, demo, input_size, focus_input_size, context_size):
        self.path = config.path_data
        self.demo = demo
        self.feature_cm = config.feature_cm
        self.feature_bp = config.feature_bp
        self.feature_maf = config.feature_maf
        self.feature_ibs = config.feature_ibs
        self.input_size, self.focus_input_size, self.context_size = input_size, focus_input_size, context_size
        self.vcf_ids = self.read_vcf_ids(self.path)
        self.sample_ids = self.read_sample_ids(config.samples_info)
        self.is_sample_in_demo = self.read_vcf_samples(self.path, self.sample_ids)
        self.num_diploids = len(self.sample_ids)
        self.num_haplotypes = 2 * self.num_diploids
        self.num_pairs = self.get_num_pairs()
        self.pair_to_haplotypes, self.haplotypes_to_pair = self.get_pair_haplotypes()
        self.num_all_variants = self.count_vcf_variants(self.path)

        # genotypes, physical positions and annotations
        self.all_phys_pos, self.genotype_matrix, self.ref_alleles,\
        self.alt_alleles, self.anc_alleles, self.rsids, self.AC, \
        self.AF, self.AC_derived, self.AF_derived = \
            self.read_vcf(self.path, self.num_haplotypes, self.num_all_variants, self.is_sample_in_demo)

        # ancestral annotations
        self.update_genotype_matrix()

        # retain polymorphic variants only (update genotype_matrix)
        self.phys_pos, self.is_polymorphic_variants, self.num_variants = self.downsample_variants()
        self.phys_dist = self.get_phys_dist(self.phys_pos).reshape(1, -1)
        # self.compute_concordant_sites()
        # self.compute_discordant_site()

        # genetic map
        genetic_map = self.read_genetic_map(config.reference_genome, demo, config.chr)
        self.gen_pos = self.get_genetic_positions(self.phys_pos, genetic_map)
        self.gen_dist = self.get_gen_dist(self.gen_pos).reshape(1, -1)

        # Compute maf
        self.maf = self.compute_maf().reshape(1, -1)
        # pdb.set_trace()

        # for __get_item__
        self.start_position = None
        self.end_position = None
        self.padding_left = None
        self.padding_right = None

    @staticmethod
    def read_vcf_ids(file):
        tot_n_samples = 0
        vcf_ids = set()
        with gzip.open(file + '.vcf.gz', 'rt') as f:
            for i, line in enumerate(f):
                if line[:6] == '#CHROM':
                    line_split = line.split()
                    for sample_id in line_split[9:]:
                        vcf_ids.add(sample_id)
                        tot_n_samples += 1
                    break
            print("Data contains " + str(tot_n_samples) + " diploid samples in total.")
        return vcf_ids

    def update_genotype_matrix(self):
        """Update genotype matrix with ancestral annotations, i.e 0 if ancestral state and 1 otherwise."""

        nb_variants = 0
        for pos in range(len(self.all_phys_pos)):
            if self.anc_alleles[pos].upper() == self.alt_alleles[pos]:
                where_0 = np.where(self.genotype_matrix[:, pos] == 0)
                where_1 = np.where(self.genotype_matrix[:, pos] == 1)
                self.genotype_matrix[where_0, pos] = 1
                self.genotype_matrix[where_1, pos] = 0
                nb_variants += 1
        print('{:.2f}% of all variants have been switched to match ancestral annotations'
              .format(100 * nb_variants / len(self.all_phys_pos)))

    def compute_concordant_sites(self):
        nb_carriers = np.sum(self.genotype_matrix, axis=0)
        nb_concordant_sites = np.sum(nb_carriers > 1)
        print('nb_concordant_sites', nb_concordant_sites)

    def compute_discordant_site(self):
        nb_carriers = np.sum(self.genotype_matrix, axis=0)
        nb_non_carriers = np.sum(1 - self.genotype_matrix, axis=0)
        nb_discordant_sites = np.sum((nb_carriers >= 1) & (nb_non_carriers >= 1))
        print('nb_discordant_sites', nb_discordant_sites)

    def downsample_variants(self):
        """Only retain polymorphic variants."""
        # variants = (self.genotype_matrix.any(axis=0)) & ~(self.genotype_matrix.all(axis=0))
        variants = self.genotype_matrix.any(axis=0)
        self.genotype_matrix = self.genotype_matrix[:, variants]
        nb_variants = self.genotype_matrix.shape[1]
        print("Data contains " + str(nb_variants) + " polymorphic variants in " + self.demo + ".")
        return self.all_phys_pos[variants], variants, nb_variants

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        (haplotype_i, haplotype_j) = self.pair_to_haplotypes[index]
        batch = dict()
        input = self.get_input(haplotype_i, haplotype_j)
        batch['input'] = torch.from_numpy(input)
        return batch

    def update_positions(self, start_position):
        if start_position - self.context_size > 0:
            self.padding_left = 0
        else:
            self.padding_left = self.context_size - start_position
        self.start_position = max(0, start_position - self.context_size)
        if start_position + self.focus_input_size + self.context_size < self.num_variants:
            self.padding_right = 0
        else:
            self.padding_right = start_position + self.focus_input_size + self.context_size - self.num_variants
        self.end_position = min(start_position + self.focus_input_size + self.context_size, self.num_variants)

    def get_input(self, haplotype_i, haplotype_j):
        """Get input for a given pair."""
        genotypes = self.genotype_matrix[[haplotype_i, haplotype_j], self.start_position: self.end_position]
        genotypes = genotypes.astype('float32')
        input = np.zeros(genotypes.shape).astype('float32')

        input[0, :] = np.logical_xor(genotypes[0, :], genotypes[1, :]).astype(float)
        input[1, :] = np.logical_and(genotypes[0, :], genotypes[1, :]).astype(float)

        if self.feature_cm:
            input = np.concatenate((input, self.gen_dist[:, self.start_position: self.end_position]))
        if self.feature_bp:
            input = np.concatenate((input, self.phys_dist[:, self.start_position: self.end_position]))
        if self.feature_maf:
            input = np.concatenate((input, self.maf[:, self.start_position: self.end_position]))
        if self.feature_ibs:
            xor_array = input[0, :]
            ibs = self.compute_ibs(xor_array)
            input = np.concatenate((input, ibs))

        # adding context to the input if necessary
        if self.padding_left > 0:
            # missing_variants = self.context_size - self.start_position
            # print("Padding on the left", missing_variants)
            input = np.pad(input, ((0, 0), (self.padding_left, 0)), 'constant')
        if self.padding_right > 0:
            # missing_variants = self.input_size - input.shape[1]
            # print("Padding on the right", missing_variants)
            input = np.pad(input, ((0, 0), (0, self.padding_right)), 'constant')

        return input.astype('float32')

    def get_pair_haplotypes(self):
        """Get mapping between pair index and haplotypes."""
        pair_to_haplotypes = {}
        haplotypes_to_pair = {}
        cpt_pair = 0
        for haplotype_i in range(0, self.num_haplotypes):
            for haplotype_j in range(haplotype_i):
                pair_to_haplotypes[cpt_pair] = (haplotype_i, haplotype_j)
                haplotypes_to_pair[(haplotype_i, haplotype_j)] = cpt_pair
                cpt_pair += 1
        assert cpt_pair == self.num_pairs, "Total number of pairs needs to be identical to the dataset size."
        return pair_to_haplotypes, haplotypes_to_pair

    def get_num_pairs(self):
        return int((self.num_haplotypes * (self.num_haplotypes - 1)) / 2)

    def compute_maf(self):
        maf = np.sum(self.genotype_matrix, axis=0) / self.genotype_matrix.shape[0]
        return np.where(maf < 0.5, maf, 1 - maf)

    def read_sample_ids(self, samples_info):
        print("Reading sample info from " + samples_info + "...")
        sample_ids = set()
        with open(samples_info, 'r') as f:
            for n_line, line in enumerate(f, 1):
                # skip header
                if n_line > 1:
                    line_split = line.split()
                    sample_id = line_split[1]
                    population = line_split[5]
                    if population == self.demo and sample_id in self.vcf_ids:
                        sample_ids.add(sample_id)
        print("Data contains " + str(len(sample_ids)) + " diploid samples in population " + self.demo + ".")
        return sample_ids

    @staticmethod
    def compute_ibs(xor_array):
        values, loc, lengths = RealDataset.island_props(xor_array)
        end_ix = 0
        ibs = np.zeros_like(xor_array)
        for i in range(len(lengths)):
            start_idx = loc[i]
            end_ix += lengths[i]
            if values[i] == 0:
                ibs[start_idx:end_ix] = lengths[i]
        return ibs.reshape(1, -1)

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
        # The values would be input array indexe by the start locations.
        # The lengths woul be the differentiation between start and stop indices.
        return v[loc], loc, np.diff(loc0)

    @staticmethod
    def get_phys_dist(phys_pos):
        """Compute physical distances from an array of physical positions."""
        prev_phys_pos = np.zeros(len(phys_pos))
        prev_phys_pos[1:] = phys_pos[:-1]
        distances = phys_pos - prev_phys_pos
        return distances

    @staticmethod
    def get_gen_dist(gen_pos):
        """Compute genetic distances from an array of genetic positions."""
        prev_gen_pos = np.zeros(len(gen_pos))
        prev_gen_pos[1:] = gen_pos[:-1]
        distances = gen_pos - prev_gen_pos
        return distances

    @staticmethod
    def get_genetic_positions(phys_pos, genetic_map):
        """Return the genetic positions."""

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
    def read_genetic_map(reference_genome, demo, chromosome):
        """Read genetic map."""

        rec_map = ''
        if reference_genome == 'hg19':
            if demo == 'constant' or demo == 'CEU.Terhorst':
                # use the average CEU_YRI_ASN rec map for constant pop size and CEU.Terhorst
                rec_map = './files/genetic-maps/average_CEU_YRI_ASN/average_CEU_YRI_ASN'
            else:
                rec_map = './files/genetic-maps/' + demo + '/' + demo
        elif reference_genome == 'hg38':
            rec_map = './files/genetic-maps-hg38/' + demo + '/' + demo

        rec_map = rec_map \
                  + '_recombination_map_hapmap_format_' + reference_genome + '_chr_' \
                  + str(chromosome) \
                  + '.txt.gz'

        # print('Computing genetic positions...')
        genetic_map = []  # list of (phys_pos, rec, gen_pos) tuples
        with gzip.open(rec_map, 'rt') as file_map:
            line = file_map.readline()
            cur_g = 0
            while line:
                line_split = line.split()
                if line_split[1] == 'position' or line[1] == '' or line_split[1] == 'Position(bp)':
                    line = file_map.readline()
                    continue
                genetic_map.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
                if cur_g > 0 and (genetic_map[cur_g][0] < genetic_map[cur_g - 1][0] or genetic_map[cur_g][2] <
                                  genetic_map[cur_g - 1][2]):
                    raise ValueError('Genetic map not in sorted order at line {}'.format(line))
                cur_g += 1
                line = file_map.readline()

        return genetic_map

    @staticmethod
    def read_sample(file, sample_ids):
        print("Reading data from " + file + '.sample' + "...")
        tot_n_samples = 0
        is_in_demo = []
        with open(file + '.sample', 'r') as f:
            for line in f:
                if line != 'ID_1 ID_2 missing' or line != '0 0 0':
                    tot_n_samples += 1
                    sample_id = line.split()[0]
                    if sample_id in sample_ids:
                        is_in_demo.append(True)
                    else:
                        is_in_demo.append(False)
        print("Data contains " + str(tot_n_samples) + " diploid samples in total.")
        return is_in_demo

    @staticmethod
    def count_hap_lines(file):
        print("Counting number of variants...")
        start_time = time()
        n_lines = 0
        with gzip.open(file + '.hap.gz', 'r') as f:
            for line in f:
                n_lines += 1
        time_duration = time() - start_time
        print("Data contains " + str(n_lines) + " variants, done in : {:.0f}min".format(time_duration / 60))
        return n_lines

    @staticmethod
    def count_vcf_variants(file):
        print("Counting number of variants...")
        start_time = time()
        n_lines = 0
        with gzip.open(file + '.vcf.gz', 'rt') as f:
            for i, line in enumerate(f):
                if line[0] != '#':
                    n_lines += 1
        time_duration = time() - start_time
        print("Data contains " + str(n_lines) + " variants, done in : {:.0f}min".format(time_duration / 60))
        return n_lines

    @staticmethod
    def read_vcf_samples(file, sample_ids):
        tot_n_samples = 0
        is_in_demo = []
        with gzip.open(file + '.vcf.gz', 'rt') as f:
            for i, line in enumerate(f):
                if line[:6] == '#CHROM':
                    line_split = line.split()
                    for sample_id in line_split[9:]:
                        tot_n_samples += 1
                        if sample_id in sample_ids:
                            is_in_demo.append(True)
                        else:
                            is_in_demo.append(False)
                    break
        return is_in_demo

    @staticmethod
    def read_vcf(file, num_haplotypes, num_variants, is_sample_in_demo):
        print("Reading data from " + file + '.vcf.gz' + "...")
        start_time = time()
        phys_pos = np.zeros(num_variants)
        genotype_matrix = np.zeros((num_haplotypes, num_variants))
        ref_alleles = num_variants * ['.']
        alt_alleles = num_variants * ['.']
        anc_alleles = num_variants * ['.']
        rsids = num_variants * ['.']
        n_samples = np.sum(is_sample_in_demo)
        AF = num_variants * ['.']
        AC = num_variants * ['.']
        AC_derived = num_variants * ['.']
        AF_derived = num_variants * ['.']
        with gzip.open(file + '.vcf.gz', 'rt') as f:
            variant = 0
            for line in f:
                if line[0] != '#':

                    line_split = line.split()
                    phys_pos[variant] = int(line_split[1])
                    rsids[variant] = line_split[2]
                    ref_alleles[variant] = line_split[3]
                    alt_alleles[variant] = line_split[4]
                    info = line_split[7]
                    alleleanc = '.'
                    if 'AA=' in info:
                        ancestral_states = info.split('AA=')[1].split(';')[0]
                        alleleanc = ancestral_states.split('|')[0]
                        # alleleanc = ancestral_states[0]
                        if '?' in alleleanc or not bool(alleleanc.strip()):
                            # invalid alleleanc
                            alleleanc = '.'
                    anc_alleles[variant] = alleleanc

                    ac = 0
                    for sample, is_in_demo in enumerate(is_sample_in_demo):
                        if is_in_demo:
                            genotype = line_split[9 + sample].split('|')
                            if genotype[0] == '1':
                                ac += 1
                            if genotype[1] == '1':
                                ac += 1
                    AF[variant] = ac / (2 * n_samples)
                    AC[variant] = ac

                    if alt_alleles[variant] == anc_alleles[variant].upper():
                        ac_derived = 2 * n_samples - ac
                    else:
                        ac_derived = ac
                    AC_derived[variant] = ac_derived
                    AF_derived[variant] = ac_derived / (2 * n_samples)

                    cpt_processed_samples = 0
                    for sample, is_in_demo in enumerate(is_sample_in_demo):
                        if is_in_demo:
                            genotype = line_split[9 + sample].split('|')
                            genotype_matrix[2 * cpt_processed_samples, variant] = genotype[0]
                            genotype_matrix[2 * cpt_processed_samples + 1, variant] = genotype[1]
                            cpt_processed_samples += 1

                    variant += 1

        ref_alleles = np.asarray(ref_alleles)
        alt_alleles = np.asarray(alt_alleles)
        anc_alleles = np.asarray(anc_alleles)
        rsids = np.asarray(rsids)
        AF = np.asarray(AF)
        AC = np.asarray(AC)
        AC_derived = np.asarray(AC_derived)
        AF_derived = np.asarray(AF_derived)
        time_duration = time() - start_time
        print("Reading data done in : {:.0f}min".format(time_duration / 60))
        return phys_pos, genotype_matrix, ref_alleles, alt_alleles, anc_alleles, rsids, AC, AF, AC_derived, AF_derived

    @staticmethod
    def read_hap(file, num_haplotypes, num_variants, is_sample_in_demo):
        print("Reading data from " + file + '.hap.gz' + "...")
        start_time = time()
        phys_pos = np.zeros(num_variants)
        genotype_matrix = np.zeros((num_haplotypes, num_variants))
        with gzip.open(file + '.hap.gz', 'r') as f:
            for variant, line in enumerate(f):
                line_split = line.split()
                phys_pos[variant] = int(line_split[1])
                cpt_processed_samples = 0
                for sample, is_in_demo in enumerate(is_sample_in_demo):
                    if is_in_demo:
                        genotype_matrix[2 * cpt_processed_samples, variant] = line_split[5 + 2 * sample]
                        genotype_matrix[2 * cpt_processed_samples + 1, variant] = line_split[5 + 2 * sample + 1]
                        cpt_processed_samples += 1
        time_duration = time() - start_time
        print("Reading data done in : {:.0f}min".format(time_duration / 60))
        return phys_pos, genotype_matrix
