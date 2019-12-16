#!/usr/bin/env python3

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
    s = sum(weights)
    thresholds = []
    t = 0
    for w in weights[:-1]:
        t += w
        thresholds.append(t / s)
    r = random.random()
    for c, t in zip(choices, thresholds):
        if r < t:
            return c
    return choices[-1]


class GenomeLayer:

    def __init__(self, genome):
        self.genome = genome

    def mutate_random(self):
        pass

    def dissimilarity(self, other):
        return self != other


class GenomeConv(GenomeLayer):

    def __init__(self, genome):
        super().__init__(genome)
        self.init_out_channels()
        self.init_activation()
        self.init_half_kernel_size()
        self.init_pool()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] +
                ', out_channels: {}, activation: {}, half_kernel_size: {}, '
                        'pool: {}'.format(
                    self.out_channels, self.activation,
                    self.half_kernel_size, self.pool) +
                r[-1:])

    def init_out_channels(self):
        self.out_channels = random.randrange(4, 16)

    def init_activation(self):
        self.activation = random.choice(['relu', 'tanh'])

    def init_half_kernel_size(self):
        self.half_kernel_size = random.randrange(3)

    def init_pool(self):
        self.pool = bool(random.randrange(2))

    def mutate_out_channels(self):
        self.out_channels = max(
                self.out_channels + (random.randrange(-2, 2) or 2), 1)

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


class GenomeLinear(GenomeLayer):

    def __init__(self, genome):
        super().__init__(genome)
        self.init_out_channels()
        self.init_activation()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] +
                ', out_channels: {}, activation: {}'.format(
                    self.out_channels, self.activation) +
                r[-1:])

    def init_out_channels(self):
        self.out_channels = random.randrange(16, 64)

    def init_activation(self):
        self.activation = random.choice(['relu', 'tanh'])

    def mutate_out_channels(self):
        self.out_channels = max(
                self.out_channels + (random.randrange(-8, 8) or 8), 1)

    def mutate_activation(self):
        if self.activation == 'relu':
            self.activation = 'tanh'
        else:
            self.activation = 'relu'

    def mutate_random(self):
        weighted_choice((
                    self.init_out_channels,
                    self.mutate_out_channels,
                    self.mutate_activation,
                ), (1, 2, 1))()

    def dissimilarity(self, other):
        return (abs(self.out_channels - other.out_channels) +
                (0 if self.activation == other.activation else 16))


class Genome:

    # WARNING:
    # There is no copy method (yet); when copying a genome, make sure to
    # explicitly create copies of each of the layers

    def __init__(self, population):
        self.population = population
        self.init_log_learning_rate()
        self.init_seed()
        self.convs = []
        self.linears = []
        self.layers_by_id = {}

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] +
                ', log_learning_rate: {}, seed: {}, convs: {}, '
                        'linears: {}'.format(
                    self.log_learning_rate, self.seed, self.convs,
                    self.linears) +
                r[-1:])

    def next_id(self):
        return self.population.next_id()

    def init_log_learning_rate(self):
        self.log_learning_rate = random.normalvariate(-6, 2)

    def init_seed(self):
        self.seed = random.randrange(2 ** 32)

    def mutate_log_learning_rate(self):
        self.log_learning_rate += random.normalvariate(0, 1)

    def add_conv(self):
        id_ = self.next_id()
        c = GenomeConv(self)
        self.convs.insert(random.randrange(len(self.convs) + 1), c)
        self.layers_by_id[id_] = c

    def remove_conv(self):
        if self.convs:
            self.convs.pop(random.randrange(len(self.convs)))

    def mutate_conv(self):
        if self.convs:
            self.convs[random.randrange(len(self.convs))].mutate_random()

    def add_linear(self):
        id_ = self.next_id()
        c = GenomeLinear(self)
        self.linears.insert(random.randrange(len(self.linears) + 1), c)
        self.layers_by_id[id_] = c

    def remove_linear(self):
        if self.linears:
            self.linears.pop(random.randrange(len(self.linears)))

    def mutate_linear(self):
        if self.linears:
            self.linears[random.randrange(len(self.linears))].mutate_random()

    def mutate_random(self):
        weighted_choice((
                    self.init_seed,
                    self.init_log_learning_rate,
                    self.mutate_log_learning_rate,
                    self.add_conv,
                    self.remove_conv,
                    self.mutate_conv,
                    self.add_linear,
                    self.remove_linear,
                    self.mutate_linear,
                ), (4, 2, 6, 4, 3, 12, 4, 3, 12))()

    def dissimilarity(self, other):
        d = (abs(self.log_learning_rate - other.log_learning_rate) * 16 +
                (0 if self.seed == other.seed else 4))
        for id_, c0 in self.layers_by_id.items():
            c1 = other.layers_by_id.get(id_)
            if c1 is not None:
                d += c0.dissimilarity(c1)
            else:
                d += 256
        for id_ in other.layers_by_id:
            if id_ not in self.layers_by_id:
                d += 256
        return d


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


# neural network components

class Net(torch.nn.Module):

    def __init__(self, genome):
        super().__init__()
        possible_activations = {
                    'relu': torch.nn.functional.relu,
                    'tanh': torch.tanh,
                }
        num_channels = 1
        width = 28
        height = 28
        self.convs = []
        for i, gconv in enumerate(genome.convs):
            conv = torch.nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=gconv.out_channels,
                        kernel_size=gconv.half_kernel_size*2+1,
                        padding=gconv.half_kernel_size,
                    )
            activation = possible_activations[gconv.activation]
            pool = gconv.pool
            self.convs.append((conv, activation, pool))
            self.add_module('convs[{}][0]'.format(i), conv)
            num_channels = gconv.out_channels
            if pool:
                width = -(-width // 2)
                height = -(-height // 2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        num_channels = num_channels * width * height
        self.linears = []
        for i, glinear in enumerate(genome.linears):
            linear = torch.nn.Linear(
                        in_features=num_channels,
                        out_features=glinear.out_channels,
                    )
            activation = possible_activations[glinear.activation]
            self.linears.append((linear, activation))
            self.add_module('linears[{}][0]'.format(i), linear)
            num_channels = glinear.out_channels
        self.final_linear = linear = torch.nn.Linear(
                    in_features=num_channels,
                    out_features=10,
                )

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        for conv, activation, pool in self.convs:
            x = conv(x)
            x = activation(x)
            if pool:
                x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))
        for linear, activation in self.linears:
            x = linear(x)
            x = activation(x)
        x = self.final_linear(x)
        return x


# create neural network from genome, then train and evaluate

def evaluate_genome_on_data(
        genome, torch_device, data_loader_train, data_loader_test):

    print('Instantiating neural network from the following genome:')
    print(genome)

    torch.random.manual_seed(genome.seed)
    net = Net(genome)
    net = net.to(torch_device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
            lr=2**genome.log_learning_rate, momentum=.9)

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


def imshow(img):
    img = img.detach().to(device='cpu').numpy()
    img = img[0]
    img = (img + 1) * (1/2)
    plt.imshow(img)
    plt.show()


def main():

    # manually seed all random number generators for reproducible results
    random.seed(0)
    #np.random.seed(0)
    torch.random.manual_seed(0)

    # set up datasets
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: x.to(device=torch_device)),
                torchvision.transforms.Normalize((1/2,), (1/2,)),
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

    ## display some test images
    #images, labels = next(iter(data_loader_test))
    #print(labels)
    #imshow(torchvision.utils.make_grid(images))

    # run genetic algorithm
    print('\n\nInitializing population\n')
    population = Population(n=16, evaluate_genome=functools.partial(
                evaluate_genome_on_data,
                torch_device=torch_device,
                data_loader_train=data_loader_train,
                data_loader_test=data_loader_test,
            ))
    for generation in range(16):
        population.evolve(generation)


if __name__ == '__main__':
    main()


