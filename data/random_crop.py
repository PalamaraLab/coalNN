import numpy as np
import random
import math


class RandomCrop(object):
    """Crop randomly the haplotypes observations in a sample.

    Args:
        input_size (float or int): Desired output size.
    """

    def __init__(self, num_variants, input_size=None, context_size=None):
        self.input_size = input_size
        self.min_start_index = -1 * context_size  # start can be negative, padding would be necessary
        self.max_start_index = max(num_variants + context_size - self.input_size, 0)  # needs to be positive

    def __call__(self):
        start = random.randint(self.min_start_index, self.max_start_index)
        return start, start + self.input_size

