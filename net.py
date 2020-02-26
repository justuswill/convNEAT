import logging
import math
import numpy as np
import torch

from gene import KernelGene, PoolGene, DenseGene
from optimizer import SGDGene, ADAMGene


def build_net_from_genome(genome, input_size, output_size):
    """
    Build net from genome, using the old weights if a elite gene or the weights of the genes (i.e. Kernel/Pool)
    Using the optimizer and hyperparameters specified in the gene
    """
    net = Net(genome, input_size=input_size, output_size=output_size)

    # Load saved parameters
    net_dict = net.state_dict()
    if genome.net_parameters is not None:
        net.load_state_dict(genome.net_parameters)
    else:
        all_net_parameters = {k: v for gene in genome.genes for k, v in gene.net_parameters.items()
                              if k in net_dict}
        net_dict.update(all_net_parameters)
        net.load_state_dict(net_dict)

    criterion = torch.nn.CrossEntropyLoss()

    opt = genome.optimizer
    if type(opt) == SGDGene:
        optimizer = torch.optim.SGD(net.parameters(), lr=2 ** opt.log_learning_rate, momentum=opt.momentum,
                                    weight_decay=2 ** opt.log_weight_decay, nesterov=True)
    elif type(opt) == ADAMGene:
        optimizer = torch.optim.Adam(net.parameters(), lr=2 ** opt.log_learning_rate,
                                     weight_decay=2 ** opt.log_weight_decay)
    else:
        raise ValueError('Optimizer %s not supported' % type(genome.optimizer))

    return net, optimizer, criterion


def train_on_data(genome, net, optimizer, criterion, epochs, torch_device, data_loader_train,
                  n_epochs_no_change=3, tol=1e-5, save_net_param=True, save_gene_param=True):
    """
    Train net
    Stop when in <n_epochs_no_change> no improvement by at least <tol> is made
    Stop when nan occurs for 5 consecutive sections (1/10 of epoch)

    Updates values in genomes that are relevant for this
    """
    net = net.to(torch_device)

    print('Beginning training')
    nan_sections = 0
    for epoch in range(epochs):
        epoch_loss = 0.
        batch_loss = 0.
        # 10 Sections
        n = len(data_loader_train) // 10
        for i, (inputs, labels) in enumerate(data_loader_train):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_loss += loss.item()
            if (i + 1) % n == 0:
                print('[{}, {:3}] loss: {:.3f}'.format(
                    epoch, i + 1, batch_loss / n))
                if np.isnan(batch_loss / n):
                    nan_sections += 1
                    if nan_sections == 5:
                        # Quit without saving net parameters
                        return
                else:
                    nan_sections = 0
                batch_loss = 0.
        epoch_loss_mean = epoch_loss / len(data_loader_train)
        print('[%d] loss: %.3f' % (epoch, epoch_loss_mean))
        genome.trained += 1
        if epoch_loss_mean < genome.loss - tol:
            genome.loss = epoch_loss_mean
            genome.no_change = 0
        else:
            genome.no_change += 1
            if genome.no_change >= n_epochs_no_change:
                # Get one epoch to improve next generation
                genome.no_change -= 1
                break
    print('Finished training')

    # Save weights and bias
    if save_gene_param:
        for name, parameter in net.state_dict().items():
            if name.startswith('conv') or name.startswith('pool_'):
                _id = int(name.split('.')[0].split('_')[-1])
                genome.genes_by_id[_id].net_parameters[name] = parameter.to('cpu')
    # Save net
    if save_net_param:
        genome.net_parameters = net.state_dict()


def evaluate(net, torch_device, data_loader_test, output_size):
    """
    Instantiate the neural network from the genome and train it for a set amount of epochs
    Evaluate the accuracy on the test data and return this as the score.
    If a monitor is set, visualize the net that is currently training.
    """
    net.to(torch_device)

    print('Beginning evaluation')
    confusion = np.zeros((output_size, output_size))
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)
            for pre, lab in zip(predictions, labels):
                confusion[lab, pre] += 1

    print(confusion)
    class_total = np.sum(confusion, axis=1)
    labeled_total = np.sum(confusion, axis=0)
    class_correct = confusion.diagonal()
    correct = int(np.sum(class_correct))
    total = int(np.sum(class_total))
    acc = correct / total

    # Ignore Warning if /0
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(output_size):
        print('%d: Recall: %5.2f %%  (%3d / %3d) - Precision %5.2f %% (%3d / %3d)' %
              (i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i],
               100 * class_correct[i] / labeled_total[i], class_correct[i], labeled_total[i]))
    print('Accuracy of the network on the %d validation images: %5.2f %% (%d / %d)' % (total, 100 * acc, correct, total))
    print()
    np.seterr(divide='warn', invalid='warn')

    return acc


class Net(torch.nn.Module):
    """
    Build a feed-forward neural net with convolutions from a genome
    """

    def __init__(self, genome, input_size, output_size):
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
        useful_genes = [gene for gene in genome.genes
                        if genome.nodes_by_id[gene.id_in].size is not None and gene.enabled]
        for gene in useful_genes:
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
                self.add_module('dense_%03d' % gene.id, dense)
                self.modules_by_id[gene.id] += [dense, possible_activations[gene.activation]]
            else:
                raise ValueError('Module type %s not supported' % type(gene))

        logging.debug('Building net Nodes')
        for node in genome.nodes:
            self.modules_by_id[node.id] = []
            if node.merge in ['upsample', 'downsample', 'avgsample']:
                self.modules_by_id[node.id] += \
                    [lambda x, size=node.target_size:
                     torch.nn.functional.interpolate(x, size=[size[1], size[2]], align_corners=False, mode='bilinear')]
            elif node.merge == 'padding':
                self.modules_by_id[node.id] += \
                    [lambda x, size=node.target_size:
                     torch.nn.ZeroPad2d([math.floor((size[2] - x.shape[3])/2), math.ceil((size[2] - x.shape[3])/2),
                                         math.floor((size[1] - x.shape[2])/2), math.ceil((size[1] - x.shape[2])/2)])(x)]
            else:
                raise ValueError('Merge type %s not supported' % node.merge)
            if node.role == 'flatten':
                self.modules_by_id[node.id] += \
                    [lambda x: torch.reshape(x, [x.shape[0], 1, 1, int(np.prod(x.shape[1:4]))])]
            if node.role == 'output':
                # Add a last layer with softmax to achieve output_size
                logging.debug('Building output Dense')
                dense = torch.nn.Linear(
                    in_features=int(np.prod(node.target_size)),
                    out_features=output_size
                )
                self.add_module('dense_out_%03d' % gene.id, dense)
                self.modules_by_id[node.id] += \
                    [lambda x: torch.reshape(x, [x.shape[0], 1, 1, int(np.prod(x.shape[1:4]))]),
                     dense,
                     lambda x: torch.reshape(x, [x.shape[0], -1])]

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
                if node.role != 'output' and list(z.shape)[1:] != node.size:
                    logging.error('output_size of node not matching: %s vs %s' % (list(z.shape[1:]), node.size))
        return outputs_by_id[2]
