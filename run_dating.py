from dating_variants.dating_msprime import MsprimeDating
import argparse
import os
import numpy as np


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    if len(memory_available) >= 1:
        return np.argmax(memory_available)
    else:
        return 0


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(get_freer_gpu())  # get most available gpu
    options = argparse.ArgumentParser(description='Model restore path.')
    options.add_argument('--restore', type=str, default='', help='Path of the model to restore weights')
    options.add_argument('--config', type=str, default='', help='Path of the config run file')
    options.add_argument('--seed', type=str, default='256', help='Seed to be used by msprime for simulation')
    options.add_argument('--size', type=str, default='', help='Sample size to be used by msprime for simulation')
    options.add_argument('--const_threshold', type=str, default='', help='Threshold for constant piecewise output')
    options.add_argument('--downsample_size', type=str, default='', help='Downsample_size for sequence data')
    options.add_argument('--ref_size', type=str, default='', help='ref_size for sequence data')
    options = options.parse_args()

    runner = MsprimeDating(options)
    runner.run()
