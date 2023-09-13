import argparse
import os
import numpy as np
from real_data.runner import Runner


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
    options.add_argument('--config', type=str, default='', help='Path to the config run file')
    options.add_argument('--restore_path', type=str, default='', help='Path of the model to restore weights')
    options.add_argument('--demo', type=str, default='', help='Population to analyse')
    options.add_argument('--output_path', type=str, default='', help='Path to output.')
    options.add_argument('--const_threshold', type=str, default='', help='Threshold for constant piecewise output')
    options.add_argument('--chr', type=str, default='', help='Chromosome to analyse')
    options.add_argument('--batch_size', type=int, default='', help='Batch size.')
    options = options.parse_args()

    runner = Runner(options)
    runner.run()