from torch.utils.data.sampler import Sampler
import random
import numpy as np
from numpy.random import random_sample
from utils import save_numpy


class Subset(Sampler):
    def __init__(self, pairs, haplotypes_to_pair):
        """
        Input:
            pairs: list of (haplotype_i, haplotype_j)
            haplotypes_to_pair: dict with key=(haplotype_i, haplotype_j), value=pair_index
        """

        self.pair_index = [haplotypes_to_pair[haps] for haps in pairs]

    def __iter__(self):
        return iter(self.pair_index)

    def __len__(self):
        return len(self.pair_index)


class ValidationThreadingSampler(Sampler):
    def __init__(self, simulations, session_name, pairs=None):
        """
        Input:
            simulations: list of simulations
            num_haplotypes: number of haplotypes to process
            session_name
            pairs: avoid ordering pairs again when restoring a session
        """

        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = self.order_pairs(simulations[0])

        save_numpy(session_name, self.pairs, 'val_ordered_pairs')

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def order_pairs(simulation):
        tree_sequence = simulation['tree_sequence']
        pair_to_haplotypes = simulation['pair_to_haplotypes']
        divergences = []
        pairs = []
        for cpt in range(len(pair_to_haplotypes)):
            idx_i, idx_j = pair_to_haplotypes[cpt]
            divergences.append(ValidationThreadingSampler.get_divergence_pair(idx_i, idx_j, tree_sequence))
            pairs.append(cpt)
        sort_index = np.argsort(divergences)
        return np.asarray(pairs)[sort_index]

    @staticmethod
    def get_divergence_pair(idx_1, idx_2, ts):
        """Compute the divergence between two haplotypes along the genome.
        """

        branch_divergence = ts.divergence([[idx_1], [idx_2]],
                                          indexes=[(0, 1)],
                                          mode='branch',
                                          windows=None,
                                          span_normalise=False)

        return branch_divergence[0]


class ThreadingSampler:
    def __init__(self, simulations, num_haplotypes, config):
        """
        Input:
            simulations: list of simulations
            num_haplotypes: number of haplotypes to process
        """

        self.simulations = simulations
        self.num_simulations = len(simulations)
        self.num_haplotypes = num_haplotypes
        # self.threshold = config.threading_threshold

        self.processed_idx = []
        for i in range(self.num_simulations):
            self.processed_idx.append([])

    def sample(self, index):
        simulation_idx = random.choice(range(self.num_simulations))

        tree_sequence = self.simulations[simulation_idx]['tree_sequence']
        ref_idx = self.processed_idx[simulation_idx]

        if len(ref_idx) == 0:
            ref_idx_1, ref_idx_2 = np.random.choice(range(self.num_haplotypes), size=2, replace=False)
            ref_idx.append(ref_idx_1)
            ref_idx.append(ref_idx_2)
            return simulation_idx, ref_idx_1, ref_idx_2

        else:
            target_idx = [i for i in range(self.num_haplotypes) if i not in ref_idx]

            if len(target_idx) == 0:
                # all haplotypes have been processed, let's re-intialise
                self.processed_idx[simulation_idx] = []
                ref_idx = self.processed_idx[simulation_idx]
                ref_idx_1, ref_idx_2 = np.random.choice(range(self.num_haplotypes), size=2, replace=False)
                ref_idx.append(ref_idx_1)
                ref_idx.append(ref_idx_2)
                return simulation_idx, ref_idx_1, ref_idx_2

            else:
                # keep processing haplotypes
                idx = random.choice(target_idx)
                closest_neighbor = self.get_min_tmrca_idx(idx, tree_sequence, ref_idx)
                ref_idx.append(idx)
                return simulation_idx, idx, closest_neighbor

    @staticmethod
    def get_min_tmrca_idx(idx, ts, ref_idx, mode='random'):
        """Compute the index of the genealogical nearest neighbor along the genome.

        This is based on the minimal TMRCA
        """

        ref_idx = np.asarray(ref_idx)
        branch_divergence = ts.divergence([[idx]] + [[i] for i in ref_idx],
                                          indexes=[(0, i) for i in range(1, ref_idx.shape[0] + 1)],
                                          mode='branch',
                                          windows=None,
                                          span_normalise=False)

        if mode == 'first':
            # This matches the first nearest neighbor found
            closest_relative_idx = ref_idx[np.argmin(branch_divergence)]
        elif mode == 'all':
            # This matches all of the closest neighbors
            closest_relative_idx = ref_idx[branch_divergence == branch_divergence.min()]
        elif mode == 'random':
            # This matches a random closest neighbor
            closest_relative_idx = ref_idx[branch_divergence == branch_divergence.min()]
            closest_relative_idx = random.choice(closest_relative_idx)

        return closest_relative_idx
