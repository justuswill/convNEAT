import functools
import itertools
import os
import pickle
import time
import random
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import AffinityPropagation, SpectralClustering
from KMedoids import KMedoids
import torch
import torchvision

from selection import cut_off_selection, tournament_selection, fitness_proportionate_selection,\
    fitness_proportionate_tournament_selection, linear_ranking_selection, stochastic_universal_sampling


def weighted_choice(choices, weights):
    return random.choices(choices, weights=weights)[0]


def random_choices(choices, chances):
    return list(np.array(choices)[np.random.rand(len(chances)) < chances])


def limited_growth(t, cap, relevance):
    k = 2.2/relevance
    return cap*(1-np.exp(-k*t))


# genetic components
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

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))

    def mutate_size(self):
        r = random.choice([-2, -1, 1, 2])
        [self.width, self.height] = map(lambda x: max(1, x + r), [self.width, self.height])

    def mutate_stride(self):
        self.stride = max(1, self.stride + random.choice([-2, -1, 1, 2]))

    def mutate_padding(self):
        self.padding = max(0, self.padding + random.choice([-2, -1, 1, 2]))

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

        out_depth = in_depth + self.depth_size_change
        out_width = ((in_width - (self.width - 1) + 2 * self.padding - 1) // self.stride) + 1
        out_height = ((in_height - (self.height - 1) + 2 * self.padding - 1) // self.stride) + 1
        return [out_depth, out_width, out_height]

    def copy(self, id=None, id_in=None, id_out=None):
        return KernelGene(id or self.id, id_in or self.id_in, id_out or self.id_out,
                          size=[self.width, self.height], stride=self.stride, padding=self.padding,
                          depth_size_change=self.depth_size_change, depth_mult=self.depth_mult)

    def dissimilarity(self, other):
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

    def mutate_height(self):
        self.height = max(1, self.height + random.choice([-2, -1, 1, 2]))

    def mutate_size(self):
        r = random.choice([-2, -1, 1, 2])
        [self.width, self.height] = map(lambda x: max(1, x + r), [self.width, self.height])

    def mutate_stride(self):
        self.stride = max(1, self.stride + random.choice([-2, -1, 1, 2]))

    def mutate_padding(self):
        # padding < half of kernel size
        self.padding = max(1, min(self.width//2, self.height//2, self.padding + random.choice([-2, -1, 1, 2])))

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
            logging.debug('Mutateted width on gene %d' % self.id)

        # force out_height > 0
        if not in_height - (self.height - 1) + 2 * self.height > 0:
            self.height = 2 * self.padding + in_height
            logging.debug('Mutateted height on gene %d' % self.id)

        out_depth = in_depth
        out_width = (in_width - (self.width - 1) + 2 * self.padding)
        out_height = (in_height - (self.height - 1) + 2 * self.padding)
        return [out_depth, out_width, out_height]

    def copy(self, id=None, id_in=None, id_out=None):
        return PoolGene(id or self.id, id_in or self.id_in, id_out or self.id_out,
                        size=[self.width, self.height], pooling=self.pooling, padding=self.padding, stride=self.stride)

    def dissimilarity(self, other):
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
        dist = np.array([self.size_change - other.size_change,
                         self.activation != other.activation])
        importance = np.array([0.6, 0.4])
        relevance = np.array([80, 0.01])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))


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
        self.max_neurons = 20000

        self.merge = merge or self.init_merge()
        self.size = None
        self.target_size = None

    def __repr__(self):
        return '<Node | ID = %d, depth=%.2f, merge=%s%s>' % (self.id, self.depth, self.merge,
                                                             ', size=%s' % str(self.size) or '')

    def short_repr(self):
        return '' if self.size is None else '%dx%dx%d' % (self.size[0], self.size[1], self.size[2])

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
        # Don't crash RAM
        if np.prod(out_size) > self.max_neurons:
            # TODO Fehler auffangen
            self.merge = 'downsample'
            logging.debug('Mutated merge on gene %d' % self.id)
            return self.output_size(in_sizes)
        self.target_size = out_size
        return [1, 1, int(np.prod(out_size))] if self.role == 'flatten' else out_size

    def copy(self):
        return Node(self.id, self.depth, merge=self.merge, role=self.role)

    def dissimilarity(self, other):
        return self.merge != other.merge


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

    def __init__(self, population, log_learning_rate=None, nodes_and_genes=None):
        self.population = population
        self.log_learning_rate = log_learning_rate or self.init_log_learning_rate()

        self.nodes, self.genes = nodes_and_genes or self.init_genome()
        self.genes_by_id, self.nodes_by_id = self.dicts_by_id()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + ' | learning_rate=%.4f, nodes=%s, genes=%s' %
                (self.log_learning_rate, self.nodes, [gene for gene in self.genes if gene.enabled]) + r[-1:])

    def init_genome(self):
        return [[Node(0, 0, role='input'), Node(1, 1, role='flatten'), Node(2, 2, role='output')],
                [Gene(3, 0, 1, mutate_to=[[KernelGene, DenseGene], [1, 0]]).mutate_random(),
                 Gene(4, 1, 2, mutate_to=[[KernelGene, DenseGene], [0, 1]]).mutate_random()]]

    def next_id(self):
        return self.population.next_id()

    def save(self):
        return [self.log_learning_rate, [(node.__class__, node.id, node.depth, node.save()) for node in self.nodes],
                [(g.__class__, g.id, g.id_in, g.id_out, g.save()) for g in self.genes]]

    def load(self, save):
        self.log_learning_rate, saved_nodes, saved_genes = save
        self.nodes = [node[0](node[1], node[2]).load(node[3]) for node in saved_nodes]
        self.genes = [g[0](g[1], g[2], g[3]).load(g[4]) for g in saved_genes]
        print(self)
        self.genes_by_id, self.nodes_by_id = self.dicts_by_id()

    def dicts_by_id(self):
        genes_by_id = dict()
        for gene in self.genes:
            genes_by_id = {**genes_by_id, **{gene.id: gene}}
        nodes_by_id = dict()
        for node in self.nodes:
            nodes_by_id = {**nodes_by_id, **{node.id: node}}
        return [genes_by_id, nodes_by_id]

    def init_log_learning_rate(self):
        return random.normalvariate(-6, 2)

    def mutate_log_learning_rate(self):
        self.log_learning_rate += random.normalvariate(0, 1)

    def mutate_genes(self, p):
        mutate = np.random.rand(len(self.genes)) < p
        for i, gene in enumerate(self.genes):
            if mutate[i]:
                self.genes[i] = gene.mutate_random()

    def mutate_nodes(self, p):
        mutate = np.random.rand(len(self.nodes)) < p
        for i, node in enumerate(self.nodes):
            if mutate[i]:
                node.mutate_random()

    def dps(self, id_s, id_t, pre=None):
        # depth first search in feed-forward net
        if id_s == id_t:
            return True
        neig = [gene.id_out for gene in self.genes if gene.id_in == id_s]
        if len(neig) == 0:
            return False

        for p in neig:
            if self.dps(p, id_t, pre=id_s):
                return True
        if pre is None:
            raise ValueError('No path through net in genome %s' % self.__repr__())
        return False

    def disable_edge(self, gene):
        """
        Tries to disable a edge
        Does nothing if no other connection to output exists.
        Returns whether deletion was successful
        """
        gene.enabled = False
        if self.dps(0, 2) is False:
            gene.enabled = True
            return False
        return True

    def mutate_disable_edge(self, tries=2):
        enabled = [gene for gene in self.genes if gene.enabled]
        if len(enabled) > 0:
            while tries > 0:
                if self.disable_edge(random.choice(enabled)):
                    return
                tries -= 1

    def enable_edge(self):
        disabled = [gene for gene in self.genes if not gene.enabled]
        if len(disabled) > 0:
            random.choice(disabled).enabled = True

    def split_edge(self):
        enabled = [gene for gene in self.genes if gene.enabled]
        if len(enabled) > 0:
            edge = random.choice(enabled)
            [d1, d2] = [self.nodes_by_id[edge.id_in].depth, self.nodes_by_id[edge.id_out].depth]
            [id1, id2, id3] = [f() for f in [self.next_id]*3]
            edge.enabled = False
            # Guarantee d1<dn<d2 and no duplicates with cut-off normalvariate
            new_node = Node(id1, min(d2+(d1+d2)/10, max(d1-(d1+d2)/10, random.normalvariate((d1 + d2) / 2, 0.0001))))
            new_edge_1 = edge.copy(id2, edge.id_in, new_node.id)
            new_edge_2 = edge.add_after(id3, new_node.id, edge.id_out)
            self.nodes += [new_node]
            self.genes += [new_edge_1, new_edge_2]
            self.nodes_by_id[id1] = new_node
            self.genes_by_id[id2] = new_edge_1
            self.genes_by_id[id3] = new_edge_2

    def add_edge(self):
        if len(self.nodes) >= 2:
            tries = 5
            while tries > 0:
                [n1, n2] = random.sample(self.nodes, 2)
                if n1.depth > n2.depth:
                    n1, n2 = n2, n1
                # only if edge doesn't exist
                if [n1.id, n2.id] in [[e.id_in, e.id_out] for e in self.genes]:
                    tries -= 1
                    continue
                id = self.next_id()
                new_edge = weighted_choice([KernelGene, PoolGene, DenseGene], [1, 1, 1])(id, n1.id, n2.id)
                self.genes += [new_edge]
                self.genes_by_id[id] = new_edge
                break

    def mutate_random(self):
        mutations = random_choices((lambda: self.mutate_genes(0.5), lambda: self.mutate_nodes(0.2),
                                    self.mutate_log_learning_rate,
                                    self.mutate_disable_edge, self.enable_edge, self.split_edge, self.add_edge),
                                   (1, 1, 0.5, 0.1, 0.1, 0.7, 0.3))
        for mutate in mutations:
            mutate()
        return self

    def visualize(self, input_size=None, dbug=False, ax=None):
        self.set_sizes(input_size)
        edgelist = ['%d %d {\'class\':\'%s\'}' % (e.id_in, e.id_out, str(type(e)).split('.')[-1][:-2])
                    for e in self.genes if e.enabled]
        G = nx.parse_edgelist(edgelist)
        edge_color_dict = {'DenseGene': 'green', 'KernelGene': 'darkorange', 'PoolGene': 'darkblue'}
        node_color_dict = {None: 'skyblue', 'flatten': 'salmon', 'input': 'turquoise', 'output': 'turquoise'}
        edge_colors = [edge_color_dict[G[u][v]['class']] for u, v in G.edges()]
        node_colors = [node_color_dict[self.nodes_by_id[int(n)].role] for n in G.nodes()]
        edge_labels = {(str(e.id_in), str(e.id_out)): e.short_repr() for e in self.genes if e.enabled}
        node_labels = {str(n.id): n.short_repr() for n in self.nodes}
        pos = self.graph_positioning()

        # if ax specified just update
        show = False
        print(ax)
        if ax is None:
            show = True
            fig, ax = plt.subplots()

        nx.draw(G, ax=ax, pos=pos, node_size=300, node_shape="s", linewidths=4, width=2,
                node_color=node_colors, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(G, ax=ax, pos=pos, edge_labels=edge_labels, font_size=8, alpha=0.9)
        if dbug:
            nx.draw_networkx_labels(G, ax=ax, pos=pos, alpha=0.7, font_size=10, font_color="dimgrey", font_weight="bold")
            nx.draw_networkx_labels(G, ax=ax, pos={n: [p[0], p[1]+0.0065] for n, p in pos.items()}, labels=node_labels,
                                    font_size=7, font_color="dimgrey", font_weight="bold")
        else:
            nx.draw_networkx_labels(G, ax=ax, pos=pos, labels=node_labels,
                                    font_size=7, font_color="dimgrey", font_weight="bold")
        if show is True:
            plt.show()
        else:
            plt.show(block=False)

    # Groups nodes by feed-forward layers
    def group_by(self):
        nodes = sorted(self.nodes, key=lambda x: x.depth)
        grouped = []
        group = []
        c = []
        for n in nodes:
            if n.id in c:
                grouped.append(group)
                group = [n]
                c = []
            else:
                group += [n]
            c += [edge.id_out for edge in self.genes if edge.id_in == n.id]
        if len(group) > 0:
            grouped.append(group)
        return grouped

    def graph_positioning(self):
        grouped_nodes = self.group_by()
        x_steps = 1 / (len(grouped_nodes) - 1)
        shift_list = [-0.03, 0, 0.03]
        pos = dict()
        for i, group in enumerate(grouped_nodes):
            shift = shift_list[i % len(shift_list)]
            x = i * x_steps
            y_list = list(np.linspace(0, 1, len(group) + 2) + shift)[1:-1]
            pos = dict(**pos, **{str(n.id): (x, y_list[j]) for j, n in enumerate(group)})
        return pos

    def set_sizes(self, input_size):
        """
        calculate the sizes of all convolutional,etc... nodes and set them for
        plotting and building the net, if no input_size is given reset every node size
        """
        for node in self.nodes:
            node.size = None
        if input_size is None:
            return
        nodes = sorted(self.nodes, key=lambda x: x.depth)
        self.nodes_by_id[0].size = input_size
        outputs_by_id = {0: input_size}
        for node in nodes:
            # All reachable incoming edges that are enabled
            in_edges = [edge for edge in self.genes if edge.enabled and edge.id_in in outputs_by_id.keys()
                        and edge.id_out == node.id]
            if len(in_edges) > 0:
                in_sizes = [edge.output_size(outputs_by_id[edge.id_in]) for edge in in_edges]
                node.size = node.output_size(in_sizes)
                outputs_by_id[node.id] = node.size

    def dissimilarity(self, other, c=[5, 5, 5, 1, 5]):
        """
        The distance/dissimilarity of two genomes, similar to NEAT
        dist = (c1*S + c2*D + c3*E)/N + c4*T + c5*K
        where
        S sum of difference in same genes
        D number of disjoint genes
        E number of excess genes
        N length of larger gene
        T difference in optimizer + hyperparameters
        K mean of difference in same nodes
        """
        genes_1, genes_2 = map(lambda x: x.genes_by_id, [self, other])
        ids_1, ids_2 = map(lambda x: set(x.keys()), [genes_1, genes_2])
        nodes_1, nodes_2 = map(lambda x: x.nodes_by_id, [self, other])
        node_ids = set(nodes_1.keys()) & set(nodes_2.keys())

        N = max(len(ids_1), len(ids_2))
        excess_start = max(ids_1 | ids_2) + 1

        S = sum([genes_1[_id].dissimilarity(genes_2[_id]) for _id in ids_1 & ids_2])
        D = len([_id for _id in ids_1 ^ ids_2 if _id < excess_start])
        E = len(ids_1 ^ ids_2) - D
        T = limited_growth(np.abs(self.log_learning_rate - other.log_learning_rate), 1, 5)
        K = sum([nodes_1[_id].dissimilarity(nodes_2[_id]) for _id in node_ids]) / len(node_ids)

        return (c[0] * S + c[1] * D + c[2] * E) / N + c[3] * T + c[4] * K


class Population:
    """
    A population of genomes to be evolved
    -----
    n               - population size
    evaluate_genome - how to get a score from genome
    genomes         - the current population of genomes
    generation      - keeps track of the current generation
    """

    def __init__(self, n, evaluate_genome, parent_selection, crossover, name=None, elitism_rate=0.05, load=None, ax=None):
        # Evolution parameters
        self.n = n
        self.evaluate_genome = evaluate_genome
        self.parent_selection = parent_selection
        self.crossover = crossover
        self.elitism_rate = elitism_rate

        # Load instead
        if load is not None:
            self.load_checkpoint(*load)
        self.ax = ax

        # Init Genomes
        self.id_generator = itertools.count()
        # 0-5 is reserved
        [f() for f in [self.next_id]*5]

        # Begin with only one species
        self.number_of_species = 1
        self.species = {0: [Genome(self) for _ in range(n)]}
        self.generation = 1

        self.checkpoint_name = name or time.strftime("%d.%m-%H:%M")

    def next_id(self):
        return next(self.id_generator)

    def save_checkpoint(self):
        save = [self.n, self.elitism_rate, self.id_generator, self.number_of_species,
                self.generation, self.checkpoint_name,
                {species: [(genome.__class__, genome.save()) for genome in genomes]
                 for species, genomes in self.species.items()}]

        _dir = os.path.join('checkpoints', self.checkpoint_name)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open(os.path.join(_dir, "%02d.cp" % self.generation), "wb") as c:
            pickle.dump(save, c)

    def load_checkpoint(self, checkpoint_name, generation):
        file_path = os.path.join('checkpoints', checkpoint_name, "%02d.cp" % generation)
        with open(file_path, "rb") as c:
            [self.n, self.elitism_rate, self.id_generator, self.number_of_species,
             self.generation, self.checkpoint_name,
             saved_genomes] = pickle.load(c)

            self.species = {species: [genome[0](self).load(genome[1]) for genome in genomes]
                            for species, genomes in saved_genomes.items()}

    def cluster(self, relative=False):
        """
        Cluster the genomes with K_Medoids-Clustering
        Change the number of species if needed (-2 .. +2)

        #absolute clustering
        The number of clusters is the minimum needed to achieve a score of <15

        # relative clustering
        The number of cluster decreases if the score of k-1 is higher by <20%
        The number of cluster decreases if the score of k+1 is smaller by >50%
        """
        k = self.number_of_species
        n = self.n
        all_genomes = [g for genomes in self.species.values() for g in genomes]

        # Distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = all_genomes[i].dissimilarity(all_genomes[j])

        # Get performance of K-Medodis for some # of clusters near k
        ids_to_check = list(range(max(1, k - 2), min(n + 1, k + 3)))
        all_labels = {i: KMedoids(n_clusters=i, metric='precomputed').fit(distances).labels_ for i in ids_to_check}
        scores = {i: self.cluster_score(distances, lab) for i, lab in all_labels.items()}
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.plot(ids_to_check, list(scores.values()))

        if relative:
            # Change number of clusters
            while k + 1 in ids_to_check and scores[k + 1] < 0.5 * scores[k]:
                print("number of clusters increased by one")
                k += 1
            while k - 1 in ids_to_check and scores[k - 1] < 1.20 * scores[k]:
                print("number of clusters increased by one")
                k -= 1
        else:
            while k + 1 in ids_to_check and scores[k] >= 15:
                print("number of clusters increased by one")
                k += 1
            while k - 1 in ids_to_check and scores[k - 1] < 15:
                print("number of clusters increased by one")
                k -= 1

        # Save new Clustering
        self.number_of_species = k
        labels = all_labels[k]
        self.species = {i: [] for i in range(k)}
        for i, g in enumerate(all_genomes):
            self.species[labels[i]] += [g]

        # Plot distance matrix after clustering
        ind = np.argsort(labels)
        ax2.imshow(distances[np.ix_(ind, ind)])
        plt.show(block=False)

    def cluster_score(self, distances, labels):
        """
        Computes the score of a k-clustering by finding the best cluster-center xi for each cluster i
        and than calculating 1/k * sum(i=0, k, sum(j in cluster i, dist(j, xi)^2))
        normalized by number of genomes in the population

        Input:
        distance metric for the n points
        labels for the n points
        """
        k = max(labels) + 1
        score = 0
        for i in range(k):
            in_cluster_distances = distances[np.ix_(labels == i, labels == i)]
            in_cluster_scores = np.sum(in_cluster_distances ** 2, axis=1)
            score += np.min(in_cluster_scores)
        return score / (k*self.n)

    def evolve(self):
        """
        Group the genomes to species and evaluate them on training data
        Generate the next generation with selection, crossover and mutation
        """
        # Saving checkpoint
        print("Saving checkpoint")
        self.save_checkpoint()

        self.cluster()
        print(self.species)
        evaluated_genomes_by_species = {species: sorted([(g, self.evaluate_genome(g, ax=self.ax)) for g in genomes],
                                                        key=lambda g: g[1], reverse=True)
                                        for species, genomes in self.species.items()}

        print('\n\nGENERATION %d\n' % self.generation)
        for species, evaluated_genomes in evaluated_genomes_by_species.items():
            print('Species %d with %d members:\n' % (species, len(evaluated_genomes)))
            for g, s in evaluated_genomes:
                r = repr(g)
                if len(r) > 64:
                    r = r[:60] + '...' + r[-1:]
                print('{:64}:'.format(r), s)
            print('\n')

        [g[0].visualize(input_size=(1, 28, 28)) for g in self.species.values()]

        for sp, evaluated_genomes in evaluated_genomes_by_species.items():
            n_sp = len(evaluated_genomes)
            elitism = math.floor(self.elitism_rate * n_sp)
            elite_genomes = [g for g, s in evaluated_genomes[:elitism]]
            parents = self.parent_selection(evaluated_genomes, k=n_sp-elitism)
            new_genomes = [self.crossover(p[0], p[1]) for p in parents]
            self.species[sp] = elite_genomes + new_genomes

        self.generation += 1


# Force outputsize and check dense layer merging
class Net(torch.nn.Module):

    def __init__(self, genome, input_size):
        super().__init__()
        possible_activations = {'relu': torch.nn.functional.relu,
                                'tanh': torch.tanh}

        genome.set_sizes(input_size)
        self.nodes = sorted(genome.nodes, key=lambda n: n.depth)
        self.modules_by_id = dict()
        # All reachable incoming edges that are enabled
        self.in_edges_by_node_id = {node.id: [edge for edge in genome.genes if edge.enabled and edge.id_out == node.id
                                              and genome.nodes_by_id[edge.id_in].size is not None]
                                    for node in self.nodes}

        logging.debug('Building net Edges')
        for gene in genome.genes:
            if genome.nodes_by_id[gene.id_in].size is not None and gene.enabled:
                self.modules_by_id[gene.id] = []
                n_in, n_out = map(lambda x: genome.nodes_by_id[x], [gene.id_in, gene.id_out])
                if type(gene) == KernelGene:
                    logging.debug('Building net Kernel %d' % gene.id)
                    depth_wise = torch.nn.Conv2d(
                        in_channels=n_in.size[0],
                        out_channels=n_in.size[0]*gene.depth_mult,
                        kernel_size=[gene.width, gene.height],
                        stride=gene.stride,
                        padding=gene.padding,
                        groups=n_in.size[0],
                        bias=False
                    )
                    point_wise = torch.nn.Conv2d(
                        in_channels=n_in.size[0]*gene.depth_mult,
                        out_channels=n_in.size[0] + gene.depth_size_change,
                        kernel_size=1,
                        bias=False
                    )
                    names = ['conv1_%03d' % gene.id, 'conv2_%03d' % gene.id]
                    self.add_module(names[0], depth_wise)
                    self.add_module(names[1], point_wise)
                    self.modules_by_id[gene.id] += [depth_wise, point_wise]
                elif type(gene) == PoolGene:
                    logging.debug('Building net Pool %d' % gene.id)
                    if gene.pooling == 'max':
                        pool = torch.nn.MaxPool2d(
                            kernel_size=[gene.width, gene.height],
                            stride=gene.stride,
                            padding=gene.padding
                        )
                    elif gene.pooling == 'avg':
                        pool = torch.nn.AvgPool2d(
                            kernel_size=[gene.width, gene.height],
                            stride=gene.stride,
                            padding=gene.padding
                        )
                    else:
                        raise ValueError('Pooling type %s not supported' % gene.pooling)
                    self.add_module('pool_%03d' % gene.id, pool)
                    self.modules_by_id[gene.id] += [pool]
                elif type(gene) == DenseGene:
                    logging.debug('Building net Dense %d' % gene.id)
                    dense = torch.nn.Linear(
                        in_features=n_in.size[2],
                        out_features=n_in.size[2] + gene.size_change
                    )
                    self.add_module('dense1_%03d' % gene.id, dense)
                    self.modules_by_id[gene.id] += [dense, possible_activations[gene.activation]]
                else:
                    raise ValueError('Module type %s not supported' % type(gene))

        logging.debug('Building net Nodes')
        for node in genome.nodes:
            self.modules_by_id[node.id] = []
            if node.merge in ['upsample', 'downsample', 'avgsample']:
                self.modules_by_id[node.id] += [lambda x, size=node.target_size:
                    torch.nn.functional.interpolate(x, size=[size[1], size[2]], align_corners=False, mode='bilinear')]
            elif node.merge == 'padding':
                self.modules_by_id[node.id] += [lambda x, size=node.target_size:
                    torch.nn.ZeroPad2d([math.floor((size[2] - x.shape[3])/2), math.ceil((size[2] - x.shape[3])/2),
                                        math.floor((size[1] - x.shape[2])/2), math.ceil((size[1] - x.shape[2])/2)])(x)]
            else:
                raise ValueError('Merge type %s not supported' % node.merge)
            if node.role == 'flatten':
                self.modules_by_id[node.id] += [lambda x:
                    torch.reshape(x, [x.shape[0], 1, 1, int(np.prod(x.shape[1:4]))])]

    def forward(self, x):
        outputs_by_id = {0: x}
        for node in self.nodes:
            # All reachable incoming edges that are enabled
            in_edges = self.in_edges_by_node_id[node.id]
            if len(in_edges) > 0:
                logging.debug(node)
                data_in = []
                # Apply Edge Genes
                for gene in in_edges:
                    y = outputs_by_id[gene.id_in]
                    logging.debug(gene)
                    logging.debug('shape before: %s' % str(y.shape))
                    for module in self.modules_by_id[gene.id]:
                        y = module(y)
                    logging.debug('shape after: %s' % str(y.shape))
                    data_in += [y]
                # node preprocessing
                logging.debug('shape before: %s' % [data.shape for data in data_in])
                data_in = list(map(self.modules_by_id[node.id][0], data_in))
                logging.debug('shape after: %s' % [data.shape for data in data_in])
                z = torch.cat(data_in, dim=1)
                # node postprocessing
                for f in self.modules_by_id[node.id][1:]:
                    z = f(z)
                outputs_by_id[node.id] = z
                if list(z.shape)[1:] != node.size:
                    logging.error('output_size of node not matching: %s vs %s' % (list(z.shape[1:]), node.size))
        return torch.reshape(outputs_by_id[2], (x.shape[0], -1))


def evaluate_genome_on_data(genome, torch_device, data_loader_train, data_loader_test, input_size, ax=None):

    print('Instantiating neural network from the following genome:')
    print(genome)
    if ax is not None:
        genome.visualize(input_size=input_size, ax=ax)
    plt.pause(1)

    logging.debug('Building Net')
    net = Net(genome, input_size=input_size)
    net = net.to(torch_device)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        optimizer = torch.optim.SGD(net.parameters(), lr=2**genome.log_learning_rate, momentum=.9)
    except ValueError:
        import pdb;pdb.set_trace()

    print('Beginning training')
    for epoch in range(2):
        epoch_loss = 0.
        batch_loss = 0.
        n = len(data_loader_train) // 10
        for i, (inputs, labels) in enumerate(data_loader_train):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_loss += loss.item()
            if (i+1) % n == 0:
                print('[{}, {:3}] loss: {:.3f}'.format(
                        epoch, i+1, batch_loss / n))
                batch_loss = 0.
        print('[{}] loss: {:.3f}'.format(
                epoch, epoch_loss / len(data_loader_train)))
    print('Finished training')

    class_total = list(0 for i in range(10))
    class_correct = list(0 for i in range(10))
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = predictions == labels
            for label, correct in zip(labels, correct_predictions):
                class_total[label] += 1
                class_correct[label] += correct.item()
    for i in range(10):
        print('Accuracy of {}: {:5.2f} %  ({} / {})'.format(
                i, 100 * class_correct[i] / class_total[i], class_correct[i],
                class_total[i]))

    total = sum(class_total)
    correct = sum(class_correct)
    print('Accuracy of the network on the {} test images: {:5.2f} %  '
          '({} / {})'.format(total, 100 * correct / total, correct, total))
    print()

    return correct / total


def crossover(genome1, genome2, more_fit_crossover_rate=0.8, less_fit_crossover_rate=0.2):
    """
    Input:
    genome1: the more fit genome
    genome2: the less fit genome
    more_fit_crossover_rate: the rate at which genes only occurring in the more fit gene are used
    less_fit_crossover_rate: -"-
    -----
    Combine the genome to get a child-genome.
    Mutate the child genome.
    """
    population = genome1.population
    child_genes = []
    child_nodes = []

    disabled_ids = []

    # Genes
    ids_1, ids_2 = map(lambda x: set(x.genes_by_id.keys()), [genome1, genome2])
    for _id in ids_1 | ids_2:
        if _id in ids_1:
            if _id in ids_2:
                child_genes += [genome1.genes_by_id[_id].copy()]
            else:
                gene = genome1.genes_by_id[_id].copy()
                if random.random() > more_fit_crossover_rate:
                    disabled_ids += [_id]
                child_genes += [gene]
        else:
            gene = genome2.genes_by_id[_id].copy()
            if random.random() > less_fit_crossover_rate:
                disabled_ids += [_id]
            child_genes += [gene]

    # Nodes
    node_ids_1, node_ids_2 = map(lambda x: set(x.nodes_by_id.keys()), [genome1, genome2])
    for _id in node_ids_1 | node_ids_2:
        if _id in node_ids_1:
            child_nodes += [genome1.nodes_by_id[_id].copy()]
        else:
            child_nodes += [genome2.nodes_by_id[_id].copy()]

    child_genome = Genome(population, log_learning_rate=genome1.log_learning_rate,
                          nodes_and_genes=[child_nodes, child_genes])

    for _id in disabled_ids:
        child_genome.disable_edge(child_genome.genes_by_id[_id])
    return child_genome.mutate_random()


def data_loader(torch_device):
    # set up datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda x: x.to(device=torch_device)),
        torchvision.transforms.Normalize((1 / 2,), (1 / 2,)),
    ])
    target_transform = torchvision.transforms.Lambda(
        lambda x: torch.tensor(x, device=torch_device))
    dataset_train = torchvision.datasets.MNIST(
        'data', train=True, transform=transform,
        target_transform=target_transform, download=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=100, shuffle=True)
    dataset_test = torchvision.datasets.MNIST(
        'data', train=False, transform=transform,
        target_transform=target_transform, download=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=100, shuffle=False)

    # peek in data to get the input_size
    peek = next(iter(data_loader_train))
    input_size = list(peek[0].shape[1:])

    return data_loader_test, data_loader_train, input_size


def main():
    # manually seed all random number generators for reproducible results
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader_test, data_loader_train, input_size = data_loader(torch_device)

    while True:
        loading = input("load from checkpoint? [y/n]")
        if loading == 'n':
            load = None
            break
        elif loading == 'y':
            checkpoint = input("checkpoint name:")
            generation = int(input("generation:"))
            load = [checkpoint, generation]
            break

    fig, ax = plt.subplots()

    print('\n\nInitializing population\n')
    p = Population(n=50, name='first_test', elitism_rate=0.05, ax=ax,
                   evaluate_genome=functools.partial(
                       evaluate_genome_on_data,
                       torch_device=torch_device,
                       data_loader_train=data_loader_train,
                       data_loader_test=data_loader_test,
                       input_size=input_size
                   ),
                   parent_selection=functools.partial(
                       fitness_proportionate_tournament_selection,
                       tournament_size=3
                   ),
                   crossover=crossover,
                   load=load)
    for _ in range(10):
        p.evolve()


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
