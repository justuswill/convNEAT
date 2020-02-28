import random
import logging
import numpy as np


class Node:
    """
    Nodes merge different incoming connections.
    Before concating a preprocessing step is performed to make the dimensions compatible
    Optionally a postprocessing step will be performed
    -----
    depth       - the depth in the network (to guarantee a feed-forward network)
    target_size - which width/height is needed after node preprocessing
    size        - the size of the output after node postprocessing
    merge       - kind of preprocessing
    role        - specific role in the net, can set the kind of postprocessing [e.g. 'flatten', 'input', 'output']
    max_neurons - won't allow more outgoing connections than this
    """

    def __init__(self, id, depth, merge=None, role=None):
        self.id = id
        self.depth = depth
        self.role = role

        self.possible_merges = ['upsample', 'downsample', 'padding', 'avgsample']
        self.max_neurons = 200000  # TODO

        self.merge = merge or self.init_merge()
        self.size = None
        self.target_size = None

    def __repr__(self):
        return '<Node | ID = %d, depth=%.2f, merge=%s%s>' % (self.id, self.depth, self.merge,
                                                             ', size=%s' % str(self.size) or '')

    def short_repr(self):
        return '' if self.target_size is None else '%dx%dx%d' % tuple(map(lambda i: self.target_size[i], [0, 1, 2]))

    def save(self):
        return [self.role, self.merge, self.size, self.target_size]

    def load(self, save):
        self.role, self.merge, self.size, self.target_size = save
        return self

    def init_merge(self):
        return random.choice(self.possible_merges)

    def mutate_merge(self):
        self.merge = random.choice([x for x in self.possible_merges if x != self.merge])

    def mutate_random(self):
        self.mutate_merge()
        return self

    def output_size(self, in_sizes):
        # how to combine the input sizes of a node
        mergesize = {'downsample': lambda x: [min([l[1] for l in x]), min([l[2] for l in x])],
                     'upsample': lambda x: [max([l[1] for l in x]), max([l[2] for l in x])],
                     'padding': lambda x: [max([l[1] for l in x]), max([l[2] for l in x])],
                     'avgsample': lambda x: [sum([l[1] for l in x]) // len(x), sum([l[2] for l in x]) // len(x)]}
        # add depths of inputs
        out_size = [sum([i[0] for i in in_sizes]), *mergesize[self.merge](in_sizes)]

        # If to much neurons use downsampling to minimize
        if np.prod(out_size) > self.max_neurons:
            self.merge = 'downsample'
            logging.debug('Mutated merge on gene %d' % self.id)
            out_size = [sum([i[0] for i in in_sizes]), *mergesize[self.merge](in_sizes)]

        self.target_size = out_size
        return [1, 1, int(np.prod(out_size))] if self.role in ['flatten', 'output'] else out_size

    def copy(self):
        return Node(self.id, self.depth, merge=self.merge, role=self.role)

    def dissimilarity(self, other):
        return self.merge != other.merge
