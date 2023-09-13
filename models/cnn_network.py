import torch.nn as nn
from collections import OrderedDict
import math
import torch
from layers.utils import RestrictionActivation


class CnnNetwork(nn.Module):
    """
    Convolutionnal neural network.
    """

    def __init__(self, focus_input_size, x_dim, config, root_time=float('inf'), clamp=False):

        super().__init__()
        self.clamp = clamp
        self.root_time = root_time

        self.run_CoalNN = False
        if hasattr(config, 'run_CoalNN') and config.run_CoalNN:
            self.run_CoalNN = True

        self.visualise_first_layer = False
        if hasattr(config, 'visualise_first_layer') and config.visualise_first_layer:
            self.visualise_first_layer = True

        self.saliency_map = False
        if hasattr(config, 'saliency_map') and config.saliency_map:
            self.saliency_map = True
        
        self.perturb_maf = False
        if hasattr(config, 'perturb_maf') and config.perturb_maf:
            self.perturb_maf = True

        n_layers = len(config.model.h_dim)
        layers = OrderedDict()

        layers['norm'] = nn.BatchNorm1d(num_features=x_dim[1])

        assert config.model.kernel_size[0] % 2 == 1
        # look at neighbors at a distance up to half of the focus_input_size
        dilation = math.ceil(focus_input_size / (2 * config.model.kernel_size[0]))
        # dilation = 1  # if no dilated convnet
        # padding = dilation * int((config.model.kernel_size[0] - 1) / 2)
        padding = 0
        layers['conv1'] = nn.Conv1d(x_dim[1], config.model.h_dim[0],
                                    kernel_size=config.model.kernel_size[0],
                                    padding=padding,
                                    dilation=dilation)
        layers['norm1'] = nn.BatchNorm1d(num_features=config.model.h_dim[0])
        layers['relu1'] = nn.ReLU()

        for layer in range(1, n_layers):
            assert config.model.kernel_size[layer - 1] % 2 == 1
            dilation = 1
            # padding = dilation * int((config.model.kernel_size[layer] - 1) / 2)
            padding = 0
            layers['conv' + str(layer + 1)] = nn.Conv1d(config.model.h_dim[layer - 1],
                                                        config.model.h_dim[layer],
                                                        kernel_size=config.model.kernel_size[layer],
                                                        padding=padding,
                                                        dilation=dilation)
            layers['norm' + str(layer + 1)] = nn.BatchNorm1d(num_features=config.model.h_dim[layer])
            layers['relu' + str(layer + 1)] = nn.ReLU()

        dilation = 1
        kernel_size = 1
        layers['conv' + str(n_layers + 1)] = nn.Conv1d(config.model.h_dim[n_layers - 1],
                                                       3,
                                                       kernel_size=kernel_size,
                                                       dilation=dilation)

        if config.model.restriction_activation:
            layers['restriction'] = RestrictionActivation(min_value=math.log(0.01),
                                                          max_value=root_time)

        self.network = nn.Sequential(layers)

    def forward(self, x):

        output = {}

        prediction = self.network(x['input'])
        if self.clamp:
            output['output'] = torch.clamp(prediction[:, 0, :], float('-inf'), self.root_time)
        else:
            output['output'] = prediction[:, 0, :]
        output['breakpoints'] = prediction[:, [1, 2], :]

        if self.visualise_first_layer:
            y = self.network[0](x['input'])
            y = self.network[1](y)
            y = self.network[2](y)
            y = self.network[3](y)
            output['hidden_layer_1_output'] = y

        if self.saliency_map:
            output['output_norm'] = self.network[0](x['input'])
            y = self.network[1:](output['output_norm'])
            pred = y[:, 0, :]
            output['saliency_pred'] = pred

        return output
