import torch.nn as nn
import torch
from functools import partial
from collections import OrderedDict


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.view(input.size(0), self.dim[-3], self.dim[-2], self.dim[-1])


class Lambda(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class ResBlock(nn.Module):
    """Residual block:
       x -> Conv -> norm -> act. -> ADD -> out
         |                           |
          ---------------------------
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1, norm='bn', activation='relu', bias=False):
        super().__init__()
        out_channels = out_channels or in_channels

        # self.layers = nn.Sequential(OrderedDict([
        #    ('conv_1', ConvBlock(in_channels, in_channels, kernel_size,
        #                         stride=1, norm=norm, activation=activation, bias=bias)),
        #    ('conv_2', ConvBlock(in_channels, out_channels, kernel_size,
        #                         stride=1, norm=norm, activation=activation, bias=bias)),
        #]))

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', ConvBlock(in_channels, out_channels, kernel_size,
                                 dilation=dilation, stride=1, norm=norm, activation=activation, bias=bias)),
        ]))

        if out_channels != in_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        x_residual = self.layers(x)

        if self.projection:
            x = self.projection(x)
        return x + x_residual


class ConvBlock(nn.Module):
    """1D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, dilation=1, norm='bn', activation='relu',
                 bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = dilation * int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d if not transpose else partial(nn.ConvTranspose1d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class RestrictionActivation(nn.Module):
    """ Constrain output to be between min_value and max_value."""

    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.scale = (max_value - min_value) / 2
        self.offset = min_value

    def forward(self, x):
        x = torch.tanh(x) + 1  # in range [0, 2]
        x = self.scale * x + self.offset  # in range [min_value, max_value]
        return x
