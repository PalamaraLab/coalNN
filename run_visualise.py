import argparse
from visualisers.visualiser_msprime import MsprimeVisualiser


if __name__ == '__main__':
    options = argparse.ArgumentParser(description='Model restore path.')
    options.add_argument('--restore', type=str, default='', help='Path of the model to restore weights')
    options.add_argument('--config', type=str, default='', help='Path of the config run file')
    options.add_argument('--seed', type=str, default='256', help='Seed to be used by msprime for simulation')
    options.add_argument('--size', type=str, default='', help='Sample size to be used by msprime for simulation')
    options.add_argument('--const_threshold', type=str, default='', help='Threshold for constant piecewise output')
    options.add_argument('--downsample_size', type=str, default='', help='Downsample_size for sequence data')
    options.add_argument('--ref_size', type=str, default='', help='ref_size for sequence data')
    options = options.parse_args()

    visualiser = MsprimeVisualiser(options)
    visualiser.visualise()
