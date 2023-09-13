import argparse
import os
import numpy as np
from trainers.trainer_msprime import MsprimeTrainer


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
    options = argparse.ArgumentParser(description='Model config/restore path.')
    options.add_argument('--config', type=str, default='',
                         help='Path of the config file')
    options.add_argument('--restore', type=str, default='',
                         help='Path of the model to restore (weights, optimiser)')
    options.add_argument('--demography', type=str, default='',
                         help='Demographic model to be used by msprime for simulation')
    options.add_argument('--val_ordered_pairs', type=str, default='',
                         help='Path to the numpy array containing TMRCA ordered pairs in validation set')
    options.add_argument('--output_path', type=str, default='',
                         help='Output path')
    options = options.parse_args()

    trainer = MsprimeTrainer(options)
    trainer.train()
