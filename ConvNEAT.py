import functools
import itertools
import operator
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


# genetic components
def weighted_choice(choices, weights):
    return np.array(choices)[np.where(np.cumsum(weights) > random.random() * np.sum(weights))[0][0]]


def random_choices(choices, chances):
    return list(np.array(choices)[np.random.rand(len(chances)) < chances])


class _Gene:

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ' | ID = %d ' % self.id + r[-1:]

    # A mutation is decided to happen. Returns itself.
    def mutate_random(self):
        return self

    def dissimilarity(self, other):
        return self != other


class KernelGene(_Gene):
    """
    Kernels are the edges of the graph
    """

    def __init__(self, id, id_in, id_out, size=[None, None, None], stride=None, padding=None):
        self.id = id
        self.id_in = id_in
        self.id_out = id_out

        [depth, width, height] = size
        self.depth = depth or self.init_depth()
        self.width = width or self.init_width()
        self.height = height or self.init_height()
        self.stride = stride or self.init_stride()
        self.padding = padding if padding is not None else self.init_padding()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size=%d*%d*%d, str=%d, pad=%d' %
                (self.depth, self.width, self.height, self.stride, self.padding) + r[-1:])

    def init_depth(self):
        return 1

    def init_width(self):
        return random.randrange(3, 6)

    def init_height(self):
        return random.randrange(3, 6)

    def init_stride(self):
        return 1

    def init_padding(self):
        return 0

    def mutate_depth(self):
        self.depth = max(1, self.depth + random.choice([-2, -1, 1, 2]))

    def mutate_width(self):
        self.width = max(1, self.width + random.choice([-2, -1, 1, 2]))

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))

    def mutate_size(self):
        r = random.choice([-2, -1, 1, 2])
        [self.width, self.height] = map(lambda x: max(1, x + r), [self.width, self.height])

    def mutate_stride(self):
        self.depth = max(1, self.depth + random.choice([-2, -1, 1, 2]))

    def mutate_padding(self):
        self.depth = max(1, self.depth + random.choice([-2, -1, 1, 2]))

    def mutate_random(self):
        mutations = random_choices((self.mutate_depth, self.mutate_width, self.mutate_height, self.mutate_size,
                                    self.mutate_stride, self.mutate_padding),
                                   (0.1, 0.1, 0.1, 0.2, 0.3, 0.2))
        for mutate in mutations:
            mutate()
        return self

class PoolGene(_Gene):
    """
    Pooling Layers are edges of the graph
    """
    def __init__(self, id, id_in, id_out, pooling=None, size=[None, None], padding=None):
        self.id = id
        self.id_in = id_in
        self.id_out = id_out

        self.possible_pooling = ['max', 'avg']

        self.pooling = pooling or self.init_pooling()
        [width, height] = size
        self.width = width or self.init_width()
        self.height = height or self.init_height()
        self.padding = padding or self.init_padding()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size=%d*%d*%d, str=%d, pad=%d' %
                (self.depth, self.width, self.height, self.stride, self.padding) + r[-1:])

    def init_pooling(self):
        self.pooling = random.choice(self.pooling)

    def init_width(self):
        return random.randrange(3, 6)

    def init_height(self):
        return random.randrange(3, 6)

    def init_padding(self):
        return 0

    def mutate_pooling(self):
        self.pooling = random.choice([x for x in self.possible_pooling if x != self.pooling])

    def mutate_width(self):
        self.width = max(1, self.width + random.choice([-2, -1, 1, 2]))

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))

    def mutate_size(self):
        [self.width, self.height] = [self.mutate_width(), self.mutate_height()]

    def mutate_padding(self):
        self.padding = max(1, self.padding + random.choice([-2, -1, 1, 2]))

    def mutate_random(self):
        mutations = random_choices((self.mutate_pooling, self.mutate_width, self.mutate_height, self.mutate_size,
                                    self.mutate_padding),
                                   (0.4, 0.2, 0.2, 0.5, 0.2))
        for mutate in mutations:
            mutate()
        return self

class DenseGene(_Gene):
    """
    Fully Conected Layers are the Edges of the rest of the graph
    We don't know the size of the flattened first layer,
    so the number of hidden neurons per layer it determined by the distance to the layer before
    """

    def __init__(self, id, id_in, id_out, size_change=None, activation=None):
        self.id = id
        self.id_in = id_in
        self.id_out = id_out

        self.possible_activations = ['relu', 'tanh']

        self.size_change = size_change or self.init_size_change()
        self.activation = activation or self.init_activation()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + 'size_change=%+d, activation=%s' %
                (self.size_change, self.activation) + r[-1:])

    def init_size_change(self):
        return random.choice(list(range(-10, -4)) + list(range(5, 11)))

    def init_activation(self):
        return random.choice(self.possible_activations)

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

class Edge(_Gene):
    """
    Symbolizes a Edge, that does nothing, only for initialization.
    Specify to what it can mutate
    """

    def __init__(self, id, id_in, id_out, mutate_to=None):
        self.id = id
        self.id_in = id_in
        self.id_out = id_out

        self.mutate_to = mutate_to or self.init_mutate_to

    def init_mutate_to(self):
        return [[KernelGene, DenseGene], [1, 1]]

    def mutate_random(self):
        return weighted_choice(*self.mutate_to)(self.id, self.id_in, self.id_out)


class Node:

    def __init__(self, id, depth):
        self.id = id
        self.depth = depth



class Genome:
    """
    Indirect representation of a feedforward convolutional net.
    This includes hyperparameters.
    The Minimal net is:
    Node 0 'Input' - Edge 3 - Node 1 'Flatten' - Edge 4 - Node 2 'Out'

    Coded as a graph where the edges are the neurons and the edges describe
    - the convolution operation (kernel)
    - the sizes of fully connected layers
    Shape and Number of neurons in a node are only decoded indirectly
    """

    def __init__(self, population, log_learning_rate=None, genes_and_nodes=None):
        self.population = population
        self.log_learning_rate = log_learning_rate or self.init_log_learning_rate()

        [self.genes, self.nodes] = genes_and_nodes or self.init_genome()
        self.genes_by_id, self.nodes_by_id = self.dicts_by_id()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + ' | learning_rate=%.4f, genes=%s' %
                (self.log_learning_rate, self.genes) + r[-1:])

    def init_genome(self):
        return [[Edge(3, 0, 1, mutate_to=[[KernelGene, DenseGene], [10, 1]]),
                 Edge(4, 1, 2, mutate_to=[[KernelGene, DenseGene], [1, 10]])],
                [Node(0, 0), Node(1, 1), Node(2, 2)]]

    def next_id(self):
        return self.population.next_id()

    def dicts_by_id(self):
        genes_by_id = dict()
        for gene in self.genes:
            genes_by_dict = {**genes_by_id, **{gene.id: gene}}
        nodes_by_id = dict()
        for node in self.nodes:
            nodes_by_dict = {**nodes_by_id, **{node.id: node}}
        return [genes_by_dict, nodes_by_dict]


    def init_log_learning_rate(self):
        return random.normalvariate(-6, 2)

    def mutate_log_learning_rate(self):
        self.log_learning_rate += random.normalvariate(0, 1)

    def mutate_genes(self, p):
        mutate = np.random.rand(len(self.genes)) < p
        for i, gene in enumerate(self.genes):
            if mutate[i]:
                self.genes[i] = gene.mutate_random()

    def mutate_random(self):
        mutations = random_choices((self.mutate_log_learning_rate, lambda: self.mutate_genes(0.5)),
                                   (0.5, 1))

        for mutate in mutations:
            mutate()
        return self

class Population:

    def __init__(self, n, evaluate_genome):
        self.evaluate_genome = evaluate_genome
        self.id_generator = itertools.count()
        self.genomes = [Genome(self) for _ in range(n)]

    def next_id(self):
        return next(self.id_generator)

    def evolve(self, generation):
        scores = [(self.evaluate_genome(g), g) for g in self.genomes]
        scores.sort(key=operator.itemgetter(0))
        print()
        print()
        print('GENERATION', generation)
        print()
        for s, g in scores:
            r = repr(g)
            if len(r) > 64:
                r = r[:60] + '...' + r[-1:]
            print('{:64}:'.format(r), s)
        print()
        print()
        self.genomes = [g for s, g in scores]
        for i in range(len(self.genomes) // 4):
            self.genomes[i] = Genome(self)
        for i in range(len(self.genomes) // 4, len(self.genomes) * 3 // 4):
            self.genomes[i].mutate_random()

def main():
    g = Genome(1)
    for i in range(10):
        print(g)
        g.mutate_random()

if __name__ == '__main__':
    main()