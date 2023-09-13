import matplotlib.pyplot as plt
from utils import get_img_from_fig, save_numpy
import torch
import torchvision.utils
import numpy as np
import os


class Plotter:

    def __init__(self, config, vis=False, tensorboard=None):
        self.asmc_vis = config.asmc_vis
        self.const_piecewise = config.const_piecewise
        self.device = torch.device('cuda') if config.gpu else torch.device('cpu')
        self.n_workers = config.n_workers
        self.path = config.session_name
        self.vis = vis
        self.tensorboard = tensorboard
        self.chr_length = config.vis_chr_length
        self.const_threshold = config.const_threshold

    def plot(self, mode, global_step, batch, output, phys_pos, asmc_output=None):
        images = self.batch_to_images(mode, batch, output, phys_pos, asmc_output)
        self.tensorboard.add_image(mode + '/plot', images, global_step)

    def visualise(self, mode, batch, output, phys_pos, asmc_output=None):
        images = self.batch_to_images(mode, batch, output, phys_pos, asmc_output)

    def batch_to_images(self, mode, batch, output, phys_pos, asmc_output=None):

        def subsample_phys_pos(index):
            return phys_pos[index]

        # move tensors to CPU memory first, so they can be converted to arrays
        ground_truth = batch['label'].to('cpu')
        pos_index = batch['pos_index'].to('cpu')
        breakpoints = batch['breakpoints'].to('cpu')
        prediction = output['output'].to('cpu')
        prediction_breakpoints = output['breakpoints'].to('cpu')

        prediction_breakpoints = prediction_breakpoints[:, 1, :]  # class 1

        phys_pos_sub = np.apply_along_axis(func1d=subsample_phys_pos, axis=0, arr=pos_index)
        n_samples = ground_truth.shape[0]

        prediction_const = n_samples * [None]
        if self.const_piecewise:
            prediction_const = output['output_const'].to('cpu')

        asmc_output_map = n_samples * [None]
        asmc_output_mean = n_samples * [None]
        if self.asmc_vis and mode != 'train':
            asmc_output_map = asmc_output['output_map'].to('cpu')
            asmc_output_mean = asmc_output['output_mean'].to('cpu')

        # save_numpy(self.path, ground_truth, 'ground_truth')
        # save_numpy(self.path, prediction, 'prediction')
        # save_numpy(self.path, prediction_breakpoints, 'prediction_breakpoints')
        # save_numpy(self.path, breakpoints, 'breakpoints')
        # save_numpy(self.path, phys_pos_sub, 'phys_pos_sub')
        # save_numpy(self.path, prediction_const, 'prediction_const')

        sample = zip(ground_truth, prediction, prediction_breakpoints, breakpoints,
                     phys_pos_sub, prediction_const, asmc_output_map, asmc_output_mean, range(n_samples))

        images = list(map(self.sample_to_image, sample))

        tensor_images = torchvision.utils.make_grid(images, nrow=n_samples, padding=0)
        return tensor_images

    def sample_to_image(self, sample):
        sample_ground_truth = np.log10(sample[0]).view(-1)
        sample_prediction = np.log10(sample[1]).view(-1)
        sample_prediction_breakpoints = sample[2].view(-1)
        sample_breakpoints = sample[3].view(-1)
        phys_pos = sample[4].reshape(-1) / 1e6  # convert bp to Mbp
        # phys_pos = np.arange(len(phys_pos))
        pair_id = sample[8]

        # only look at first chr_length Mbp
        max_index = next(x[0] for x in enumerate(phys_pos) if x[1] > self.chr_length)
        sample_ground_truth = sample_ground_truth[:max_index]
        sample_prediction = sample_prediction[:max_index]
        sample_prediction_breakpoints = sample_prediction_breakpoints[:max_index]
        sample_breakpoints = sample_breakpoints[:max_index]
        phys_pos = phys_pos[:max_index]

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        # upper figure
        ax1.tick_params(labelsize=8)
        l1, = ax1.step(phys_pos, sample_ground_truth, 'forestgreen', label='Ground truth', where='post')
        l2, = ax1.step(phys_pos, sample_prediction, 'darkturquoise', linestyle='--', label='Prediction', where='post')
        line_labels = ['Ground truth',
                       'Prediction']
        lines = [l1, l2]
        if self.asmc_vis:
            if sample[6] is not None and sample[7] is not None:
                asmc_prediction_map = np.log10(sample[6]).view(-1)
                asmc_prediction_mean = np.log10(sample[7]).view(-1)
                asmc_prediction_map = asmc_prediction_map[:max_index]
                asmc_prediction_mean = asmc_prediction_mean[:max_index]
                l_asmc_map, = ax1.step(phys_pos, asmc_prediction_map, 'mediumseagreen', linestyle='--', label='ASMC MAP')
                l_asmc_mean, = ax1.step(phys_pos, asmc_prediction_mean, 'darkgreen', linestyle='--', label='ASMC mean')
                line_labels.append('ASMC MAP')
                line_labels.append('ASMC mean')
                lines.append(l_asmc_map)
                lines.append(l_asmc_mean)
        if self.const_piecewise:
            if sample[5] is not None:
                sample_prediction_const = np.log10(sample[5]).view(-1)
                sample_prediction_const = sample_prediction_const[:max_index]
                l3, = ax1.step(phys_pos, sample_prediction_const, 'royalblue',
                               label='Piecewise constant prediction', where='post')
                line_labels.append('Piecewise constant prediction')
                lines.append(l3)
        # Create axis for upper figure
        ax1.tick_params(labelsize=9)
        ax1.get_xaxis().set_visible(False)
        ymin_tick = int(sample_ground_truth.min())
        ymax_tick = int(sample_ground_truth.max()) + 1
        ticks = range(ymin_tick, ymax_tick + 1, 1)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(['$10^{' + str(x) + '}$' for x in ticks])
        ax1.set_ylabel('TMRCA \n (generations)', fontsize=11, multialignment='center')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.get_yaxis().set_label_coords(-0.08, 0.5)

        # bottom figure
        ax2.vlines(phys_pos[sample_breakpoints == 1], ymin=0,
                   ymax=1, colors='forestgreen',
                   linewidth=1.35)
        ax2.vlines(phys_pos, ymin=0, ymax=sample_prediction_breakpoints,
                   linestyle='--', color='darkturquoise', linewidth=1.35)

        l4 = ax2.axhline(y=self.const_threshold, color='firebrick', label='User-specified threshold')
        line_labels.append('User-specified threshold')
        lines.append(l4)
        # Create axis for bottom figure
        ax2.tick_params(labelsize=9)
        ax2.set_xlabel('Genomic site (Mbp)', fontsize=11)
        ax2.set_ylabel('Probability of \n recombination', fontsize=11, multialignment='center')
        ax2.set_ylim(0, 1)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.get_yaxis().set_label_coords(-0.08, 0.5)

        # Create the legend
        fig.legend(handles=lines,
                   labels=line_labels,  # The labels for each line
                   loc="upper center",  # Position of legend
                   ncol=4,
                   columnspacing=0.55,
                   frameon=False,
                   fontsize=11)

        data = get_img_from_fig(fig, dpi=180)

        if self.vis:
            plt.savefig(os.path.join(self.path, 'vis_tmrca_pair_' + str(pair_id) + '.pdf'),
                        format='pdf',
                        bbox_inches='tight')

        plt.close()

        return torch.tensor(np.transpose(data, (2, 0, 1)))
