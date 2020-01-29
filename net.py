import logging
import math
import numpy as np
import torch

from gene import KernelGene, PoolGene, DenseGene


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
