import random
import logging
import numpy as np

from tools import weighted_choice, random_choices, limited_growth


class Gene:
    """
    Symbolizes a Edge in the Graph
    Is also used in initialization.
    Specify what can be created after this in the split_edge mutation
    (or to what it will be changed if its a initializing edge)
    """

    def __init__(self, id, id_in, id_out, mutate_to=None):
        self.id = id
        self.id_in = id_in
        self.id_out = id_out
        self.enabled = True

        self.mutate_to = mutate_to or self.init_mutate_to()

        # Weights & bias - To be set after first training.
        self.net_parameters = dict()

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ' |%s| ID = %d (%d->%d) ' % (self.enabled, self.id, self.id_in, self.id_out) + r[-1:]

    def short_repr(self):
        return ''

    def save(self):
        pass

    def load(self, save):
        return self

    def init_mutate_to(self):
        return [[KernelGene, DenseGene], [1, 1]]

    # A mutation is decided to happen. Returns itself.
    def mutate_random(self):
        return weighted_choice(*self.mutate_to)(self.id, self.id_in, self.id_out)

    def output_size(self, input_size):
        pass

    # A Edge is decided to be added after. Returns what it should be.
    def add_after(self, id, id_in, id_out):
        return weighted_choice(*self.mutate_to)(id, id_in, id_out)

    def copy(self, id, id_in, id_out):
        raise ValueError("Not intended to copy")

    # How similiar are the genes between 0 (same) and 1 (very different)
    def dissimilarity(self, other):
        return self != other


class KernelGene(Gene):
    """
    Kernels are the edges of the graph
    """

    def __init__(self, id, id_in, id_out, size=[None, None], stride=None, padding=None,
                 depth_size_change=None, depth_mult=None):
        super().__init__(id, id_in, id_out, mutate_to=self.init_mutate_to())

        width, height = size
        self.width = width or self.init_width()
        self.height = height or self.init_height()
        self.stride = stride or self.init_stride()
        self.padding = padding if padding is not None else self.init_padding()
        self.depth_size_change = depth_size_change if depth_size_change is not None else self.init_depth_size_change()
        self.depth_mult = depth_mult or self.init_depth_mult()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size=%d*%d, depth_change=%d, str=%d, pad=%d, d_mult=%d' %
                (self.width, self.height, self.depth_size_change, self.stride, self.padding, self.depth_mult) + r[-1:])

    def short_repr(self):
        return '%dx%d' % (self.width, self.height)

    def save(self):
        return [self.width, self.height, self.stride, self.padding, self.depth_size_change, self.depth_mult,
                self.enabled]

    def load(self, save):
        self.width, self.height, self.stride, self.padding, self.depth_size_change, self.depth_mult, self.enabled = save
        return self

    def init_width(self):
        return random.randrange(3, 6)

    def init_height(self):
        return random.randrange(3, 6)

    def init_stride(self):
        return 1

    def init_padding(self):
        return 0

    def init_depth_size_change(self):
        return 0

    def init_depth_mult(self):
        return 1

    def init_mutate_to(self):
        return [[KernelGene, PoolGene, DenseGene], [1, 2, 0]]

    def mutate_width(self):
        self.width = max(1, self.width + random.choice([-2, -1, 1, 2]))
        # force padding <= half of kernel size
        if 2 * self.padding > self.width:
            self.padding = self.width // 2

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))
        # force padding <= half of kernel size
        if 2 * self.padding > self.height:
            self.padding = self.height // 2

    def mutate_size(self):
        r = random.choice([-2, -1, 1, 2])
        [self.width, self.height] = map(lambda x: max(1, x + r), [self.width, self.height])
        # force padding <= half of kernel size
        if 2 * self.padding > min(self.width, self.height):
            self.padding = min(self.width // 2, self.height // 2)

    def mutate_stride(self):
        self.stride = max(1, self.stride + random.choice([-2, -1, 1, 2]))

    def mutate_padding(self):
        self.padding = max(0, min(self.width // 2, self.height // 2, self.padding + random.choice([-2, -1, 1, 2])))

    def mutate_depth_size_change(self):
        self.depth_size_change = self.depth_size_change + random.choice([-2, -1, 1, 2])

    def mutate_depth_mult(self):
        self.depth_mult = max(1, self.depth_mult + random.choice([-2, -1, 1, 2]))

    def mutate_random(self):
        mutations = random_choices((self.mutate_width, self.mutate_height, self.mutate_size,
                                    self.mutate_stride, self.mutate_padding,
                                    self.mutate_depth_size_change, self.mutate_depth_mult),
                                   (0.1, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1))
        for mutate in mutations:
            mutate()
        return self

    def output_size(self, in_size):
        [in_depth, in_width, in_height] = in_size

        # force out_depth > 0
        if not in_depth + self.depth_size_change > 0:
            self.depth_size_change = 1 - in_depth
            logging.debug('Mutated depth_size_change on gene %d' % self.id)

        # force out_width > 0
        if not in_width - (self.width - 1) + 2 * self.padding > 0:
            self.width = 2 * self.padding + in_width
            logging.debug('Mutated width on gene %d' % self.id)

        # force out_height > 0
        if not in_height - (self.height - 1) + 2 * self.height > 0:
            self.height = 2 * self.padding + in_height
            logging.debug('Mutated height on gene %d' % self.id)

        # force padding <= half of kernel size
        if not 2 * self.padding <= min(self.width, self.height):
            self.padding = min(self.width // 2, self.height // 2)

        out_depth = in_depth + self.depth_size_change
        out_width = ((in_width - (self.width - 1) + 2 * self.padding - 1) // self.stride) + 1
        out_height = ((in_height - (self.height - 1) + 2 * self.padding - 1) // self.stride) + 1
        return [out_depth, out_width, out_height]

    def copy(self, id=None, id_in=None, id_out=None):
        return KernelGene(id or self.id, id_in or self.id_in, id_out or self.id_out,
                          size=[self.width, self.height], stride=self.stride, padding=self.padding,
                          depth_size_change=self.depth_size_change, depth_mult=self.depth_mult)

    def dissimilarity(self, other):
        if not isinstance(other, self.__class__):
            return 1
        dist = np.array([self.height - other.height, self.width - other.width,
                         self.stride - other.stride, self.padding - other.padding,
                         self.depth_size_change - other.depth_size_change, self.depth_mult - other.depth_mult])
        importance = np.array([0.2, 0.2, 0.1, 0.05, 0.1, 0.35])
        relevance = np.array([5, 5, 3, 3, 5, 8])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))


class PoolGene(Gene):
    """
    Pooling Layers are edges of the graph
    """
    def __init__(self, id, id_in, id_out, pooling=None, size=[None, None], stride=None, padding=None):
        super().__init__(id, id_in, id_out, mutate_to=self.init_mutate_to())

        self.possible_pooling = ['max', 'avg']

        self.pooling = pooling or self.init_pooling()
        [width, height] = size
        self.width = width or self.init_width()
        self.height = height or self.init_height()
        self.stride = stride or self.init_stride()
        self.padding = padding if padding is not None else self.init_padding()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'pool=%s, size=%d*%d, str=%d, pad=%d' %
                (self.pooling, self.width, self.height, self.stride, self.padding) + r[-1:])

    def short_repr(self):
        return self.pooling

    def save(self):
        return [self.pooling, self.width, self.height, self.stride, self.padding, self.enabled]

    def load(self, save):
        self.pooling, self.width, self.height, self.stride, self.padding, self.enabled = save
        return self

    def init_pooling(self):
        return random.choice(self.possible_pooling)

    def init_width(self):
        return random.randrange(3, 6)

    def init_height(self):
        return random.randrange(3, 6)

    def init_stride(self):
        return 1

    def init_padding(self):
        return 0

    def init_mutate_to(self):
        return [[KernelGene, PoolGene, DenseGene], [4, 1, 0]]

    def mutate_pooling(self):
        self.pooling = random.choice([x for x in self.possible_pooling if x != self.pooling])

    def mutate_width(self):
        self.width = max(1, self.width + random.choice([-2, -1, 1, 2]))
        # force padding <= half of kernel size
        if 2 * self.padding > self.width:
            self.padding = self.width // 2

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))
        # force padding <= half of kernel size
        if 2 * self.padding > self.height:
            self.padding = self.height // 2

    def mutate_size(self):
        r = random.choice([-2, -1, 1, 2])
        [self.width, self.height] = map(lambda x: max(1, x + r), [self.width, self.height])
        # force padding <= half of kernel size
        if 2 * self.padding > min(self.width, self.height):
            self.padding = min(self.width // 2, self.height // 2)

    def mutate_stride(self):
        self.stride = max(1, self.stride + random.choice([-2, -1, 1, 2]))

    def mutate_padding(self):
        # force padding <= half of kernel size
        self.padding = max(0, min(self.width//2, self.height//2, self.padding + random.choice([-2, -1, 1, 2])))

    def mutate_random(self):
        mutations = random_choices((self.mutate_pooling, self.mutate_width, self.mutate_height, self.mutate_size,
                                    self.mutate_stride, self.mutate_padding),
                                   (0.4, 0.2, 0.2, 0.5, 0.2, 0.2))
        for mutate in mutations:
            mutate()
        return self

    def output_size(self, in_size):
        [in_depth, in_width, in_height] = in_size

        # force out_width > 0
        if not in_width - (self.width - 1) + 2 * self.padding > 0:
            self.width = 2 * self.padding + in_width
            logging.debug('Mutated width on gene %d' % self.id)

        # force out_height > 0
        if not in_height - (self.height - 1) + 2 * self.height > 0:
            self.height = 2 * self.padding + in_height
            logging.debug('Mutated height on gene %d' % self.id)

        # force padding <= half of kernel size
        if 2 * self.padding > min(self.width, self.height):
            self.padding = min(self.width // 2, self.height // 2)
            logging.debug('Mutated padding on gene %d' % self.id)

        out_depth = in_depth
        out_width = (in_width - (self.width - 1) + 2 * self.padding)
        out_height = (in_height - (self.height - 1) + 2 * self.padding)
        return [out_depth, out_width, out_height]

    def copy(self, id=None, id_in=None, id_out=None):
        return PoolGene(id or self.id, id_in or self.id_in, id_out or self.id_out,
                        size=[self.width, self.height], pooling=self.pooling, padding=self.padding, stride=self.stride)

    def dissimilarity(self, other):
        if not isinstance(other, self.__class__):
            return 1
        dist = np.array([self.height - other.height, self.width - other.width,
                         self.stride - other.stride, self.padding - other.padding,
                         self.pooling != other.pooling])
        importance = np.array([0.2, 0.2, 0.1, 0.1, 0.4])
        relevance = np.array([5, 5, 3, 3, 0.01])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))


class DenseGene(Gene):
    """
    Fully Conected Layers are the Edges of the rest of the graph
    We don't know the size of the flattened first layer,
    so the number of hidden neurons per layer it determined by the distance to the layer before
    """

    def __init__(self, id, id_in, id_out, size_change=None, activation=None):
        super().__init__(id, id_in, id_out, mutate_to=self.init_mutate_to())

        self.possible_activations = ['relu', 'tanh']

        self.size_change = size_change or self.init_size_change()
        self.activation = activation or self.init_activation()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size_change=%+d, activation=%s' %
                (self.size_change, self.activation) + r[-1:])

    def save(self):
        return [self.size_change, self.activation, self.enabled]

    def load(self, save):
        self.size_change, self.activation, self.enabled = save
        return self

    def init_size_change(self):
        return random.choice(list(range(-10, -4)) + list(range(5, 11)))

    def init_activation(self):
        return random.choice(self.possible_activations)

    def init_mutate_to(self):
        return [[KernelGene, PoolGene, DenseGene], [0, 0, 1]]

    def mutate_size_change(self):
        self.size_change = self.size_change + random.choice(list(range(-10, -4)) + list(range(5, 11)))

    def mutate_activation(self):
        self.activation = random.choice([x for x in self.possible_activations if x != self.activation])

    def mutate_random(self):
        mutations = random_choices((self.mutate_size_change, self.mutate_activation),
                                   (1, 0.2))
        for mutate in mutations:
            mutate()
        return self

    def output_size(self, in_size):
        # force out_depth > 0
        if in_size[0] + self.size_change <= 0:
            self.size_change = 1 - in_size[0]
            logging.debug('Mutateted size_change on gene %d' % self.id)

        return [in_size[0], in_size[1], in_size[2] + self.size_change]

    def copy(self, id=None, id_in=None, id_out=None):
        return DenseGene(id or self.id, id_in or self.id_in, id_out or self.id_out,
                         size_change=self.size_change, activation=self.activation)

    def dissimilarity(self, other):
        if not isinstance(other, self.__class__):
            return 1
        dist = np.array([self.size_change - other.size_change,
                         self.activation != other.activation])
        importance = np.array([0.6, 0.4])
        relevance = np.array([80, 0.01])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))