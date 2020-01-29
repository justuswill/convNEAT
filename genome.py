import random
import networkx as nx

from tools import weighted_choice, random_choices, limited_growth
from node import Node
from gene import Gene, KernelGene, PoolGene, DenseGene


class Genome:
    """
    Indirect representation of a feed-forward convolutional net.
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
        self.genes_by_id, self.nodes_by_id = self.dicts_by_id()
        return self

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

    def visualize(self, ax, input_size=None, dbug=False):
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