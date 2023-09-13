import os
import sys
import io
import cv2
import numpy as np
from argparse import Namespace
import time
import os
import psutil
import yaml
import torch
from matplotlib import pyplot
from sklearn.neighbors import KernelDensity


class Config(Namespace):
    def __init__(self, config):
        super().__init__()
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def update_config(self, config_run):
        for key, value in config_run.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)

    @staticmethod
    def update_config_file(session_name, key, value):
        with open(os.path.join(session_name, 'config.yml'), 'r') as f:
            config = yaml.safe_load(f)
        config[key] = value
        with open(os.path.join(session_name, 'config.yml'), 'w') as f:
            yaml.safe_dump(config, f)


class Logger(object):
    """Save terminal outputs to log file, and continue to print on the terminal."""

    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log = open(log_filename, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        pass

    def close(self):
        self.log.flush()
        os.fsync(self.log.fileno())
        self.log.close()


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result

    return wrapper


def format_time(s):
    """Convert time in seconds to time in hours, minutes and seconds."""
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}h{m:02d}m{s:02d}s'


def print_model_spec(model, name=''):
    n_parameters = count_n_parameters(model)
    n_trainable_parameters = count_n_parameters(model, only_trainable=True)
    print(f'Model {name}: {n_parameters:.2f}M parameters of which {n_trainable_parameters:.2f}M are trainable.\n')
    return n_parameters, n_trainable_parameters


def count_n_parameters(model, only_trainable=False):
    if only_trainable:
        n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        n_parameters = sum([p.numel() for p in model.parameters()])
    return n_parameters / 10 ** 6


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_numpy(path, array, array_name):
    """Save array to disk."""
    path_file = os.path.join(path, array_name + '.npy')
    np.save(path_file, array)


def load_numpy(path, array_name):
    """Load array to disk."""
    path_file = os.path.join(path, array_name + '.npy')
    return np.load(path_file)


def save_torch(path, tensor, tensor_name):
    path_file = os.path.join(path, tensor_name + '.pt')
    torch.save(tensor, path_file)


def plot_TMRCA_pred_histogram(hom_TMRCAs, het_TMRCAs, session_name):
    subsample_hom = np.random.choice(len(hom_TMRCAs), size=35000, replace=False)
    hom_vals = hom_TMRCAs[subsample_hom]

    subsample_het = np.random.choice(len(het_TMRCAs), size=35000, replace=False)
    het_vals = het_TMRCAs[subsample_het]

    fig = pyplot.figure()
    axs = pyplot.gca()

    ymin = 1
    ymax = 6
    ticks = range(ymin, ymax + 1, 1)

    nb_samples = len(hom_vals)
    num_bins = 50
    bins = np.linspace(1, 6, num=num_bins)

    X_hom = np.random.choice(np.log10(hom_vals),
                             size=nb_samples,
                             replace=False)
    X_het = np.random.choice(np.log10(het_vals),
                             size=nb_samples,
                             replace=False)

    axs.hist(X_hom, bins=bins, density=True, alpha=0.2, label='Homozygous sites', color='green')
    axs.hist(X_het, bins=bins, density=True, alpha=0.2, label='Heterozygous sites', color='C0')
    axs.set_xticks([t for t in ticks])
    axs.tick_params(axis="x", labelsize=12)
    axs.set_xticklabels(['$10^{' + str(x) + '}$' for x in ticks])
    axs.set_yticks([])
    axs.tick_params(axis="y", labelsize=12)
    axs.set_xlabel('Pairwise TMRCA', fontsize=13.5)
    # axs.set_title('Uniform sampler', fontsize=14)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_visible(False)

    # axs.set_ylabel('Frequence', fontsize=13.5)
    axs.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1.1))

    # instantiate and fit the KDE model
    X = X_hom.reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.25, kernel='gaussian')
    kde.fit(X)
    # score_samples returns the log of the probability density
    logprob_uniform = kde.score_samples(bins.reshape(-1, 1))
    axs.plot(bins, np.exp(logprob_uniform), alpha=1, c='green')

    # instantiate and fit the KDE model
    X = X_het.reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.25, kernel='gaussian')
    kde.fit(X)
    # score_samples returns the log of the probability density
    logprob_threading = kde.score_samples(bins.reshape(-1, 1))
    axs.plot(bins, np.exp(logprob_threading), c='C0', alpha=1)

    fig.savefig(session_name + '/predicted_TMRCAs_histogram.jpg', dpi=1080)
