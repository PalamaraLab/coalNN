import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))
        self.log_tmrca = config.log_tmrca
        self.weighted_loss = config.weighted_loss

    def forward(self, batch):

        output = self.model(batch)

        if self.weighted_loss:
            huber_loss = self.weighted_huber_loss(batch['label'], output['output'])
        else:
            huber_loss = self.huber_loss(batch['label'], output['output'])

        precision1 = torch.exp(-self.log_vars[0])
        loss = 1 / 2 * precision1 * huber_loss + 1 / 2 * self.log_vars[0]

        breakpoints_loss = self.cross_entropy_with_logits(output['breakpoints'], batch['breakpoints'])
        precision2 = torch.exp(-self.log_vars[1])
        loss += precision2 * breakpoints_loss + 1 / 2 * self.log_vars[1]

        # loss is (N, 1)
        loss = torch.mean(loss)

        return loss, output, self.log_vars.data.tolist()

    @staticmethod
    def cross_entropy_weight(data):
        unique, counts = torch.unique(data, sorted=True, return_counts=True)
        if torch.numel(unique) == 1:
            # there is no recombination event in this batch
            return None
        counts = (1. / torch.sum(counts).item()) * counts
        weight = counts.pow_(-1)
        weight = (1. / torch.sum(weight)) * weight
        return weight

    @staticmethod
    def cross_entropy_with_logits(output, label):
        weight = MultiTaskLossWrapper.cross_entropy_weight(label)
        return torch.mean(F.cross_entropy(output, label, reduction='none', weight=weight), dim=-1)

    @staticmethod
    def huber_loss(label, output):
        return torch.mean(F.smooth_l1_loss(output, label, reduction='none'), dim=-1)

    def regression_weight(self, label):
        # data is expected to be in log scale
        if self.log_tmrca:
            data = torch.exp(label)
        data = torch.log10(data)
        data = torch.max(data, torch.zeros_like(data))
        min_value = data.min().int()
        max_value = data.max().int() + 1
        weights = torch.histc(data, min=min_value, max=max_value, bins=max_value - min_value)
        weights = 1. / weights
        bins_allocations = data.long() - min_value
        data_weights = weights[bins_allocations]
        data_weights = torch.nn.functional.normalize(data_weights, p=1, dim=-1)
        return data_weights

    @staticmethod
    def weighted_l1_loss(distance, weight, beta=1):
        return weight * (distance - 0.5 * beta)

    @staticmethod
    def weighted_l2_loss(difference, weight, beta=1):
        return weight * (0.5 * (difference ** 2) / beta)

    def weighted_huber_loss(self, label, output, beta=1):
        weight = self.regression_weight(label)
        difference = label - output
        distance = torch.abs(difference)
        l1_loss = MultiTaskLossWrapper.weighted_l1_loss(distance, weight, beta)
        l2_loss = MultiTaskLossWrapper.weighted_l2_loss(difference, weight, beta)
        huber_loss = torch.where(distance < beta, l2_loss, l1_loss)
        return torch.sum(huber_loss, dim=-1)
