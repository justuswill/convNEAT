import functools
import itertools
import operator
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


class _Gene:

    def mutate_random(self):
        pass

    def dissimilarity(self, other):
        return self != other


class KernelGene(_Gene):
    '''
    Kernels are the edges of the Convolutional part of the graph
    '''

    def __init__(self, id_in, id_out, size=[None, None, None] , stride=None, padding=None):

        self.id_in = id_in
        self.id_out = id_out

        [depth, width, height] = size
        self.init_depth(depth)
        self.init_width(width)
        self.init_height(height)
        self.init_stride(stride)
        self.init_padding(padding)

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size=%d*%d*%d, str=%d, pad=%d' %
                (self.depth, self.width, self.height, self.stride, self.padding) + r[-1:])

    def init_depth(self, depth):
        if depth is None:
            depth = 1
        self.depth = depth

    def init_width(self, width):
        if width is None:
            depth = 5
        self.width = width

    def init_height(self, height):
        if height is None:
            height = 1
        self.height = height

    def init_stride(self, stride):
        if stride is None:
            depth = 1
        self.stride = stride

    def mutate_activation(self):
        if self.activation == 'relu':
            self.activation = 'tanh'
        else:
            self.activation = 'relu'

    def mutate_half_kernel_size(self):
        self.half_kernel_size = max(
                self.half_kernel_size + (random.randrange(-1, 1) or 1), 0)

    def mutate_pool(self):
        self.pool = not self.pool

    def mutate_random(self):
        weighted_choice((
                    self.init_out_channels,
                    self.init_half_kernel_size,
                    self.mutate_out_channels,
                    self.mutate_activation,
                    self.mutate_half_kernel_size,
                    self.mutate_pool,
                ), (1, 1, 2, 1, 2, 1))()

    def dissimilarity(self, other):
        return (abs(self.out_channels - other.out_channels) * 4 +
                (0 if self.activation == other.activation else 16) +
                abs(self.half_kernel_size - other.half_kernel_size) * 8 +
                (0 if self.pool == other.pool else 32))