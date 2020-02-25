import functools
import random
import numpy as np
import logging
import os

import torch
import torchvision

from population import Population
from selection import cut_off_selection, tournament_selection, fitness_proportionate_selection,\
    fitness_proportionate_tournament_selection, linear_ranking_selection, stochastic_universal_sampling
from crossover import crossover
from net import train_on_data, evaluate
from monitor import Monitor
from exploration import show_genomes, from_human_readable


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
    train_val = torchvision.datasets.MNIST(
        'data', train=True, transform=transform,
        target_transform=target_transform, download=True)
    dataset_train, dataset_val = torch.utils.data.random_split(
        train_val, [int(0.85 * len(train_val)), int(0.15 * len(train_val))])
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=100, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=100, shuffle=True)
    dataset_test = torchvision.datasets.MNIST(
        'data', train=False, transform=transform,
        target_transform=target_transform, download=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=100, shuffle=False)

    # peek in data to get the input_size
    peek = next(iter(data_loader_train))
    input_size = list(peek[0].shape[1:])
    output_size = 10

    return data_loader_test, data_loader_train, data_loader_val, input_size, output_size


def main():
    # manually seed all random number generators for reproducible results
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader_test, data_loader_train, data_loader_val, input_size, output_size = data_loader(torch_device)

    # Load from checkpoint?
    while True:
        loading = input("load from checkpoint? [y/n]")
        if loading == 'e':
            show_genomes(input_size=input_size)
            return
        if loading == 'f':
            from_human_readable(input_size=input_size, output_size=output_size,
                                evaluate=functools.partial(
                                    evaluate,
                                    torch_device=torch_device,
                                    data_loader_test=data_loader_val,
                                    output_size=output_size
                                ))
            return
        elif loading == 'n':
            load = None
            break
        elif loading == 'y':
            checkpoint = input("checkpoint name:")
            generation = int(input("generation:"))
            if not os.path.exists(os.path.join("checkpoints", checkpoint)):
                print("This checkpoint doesn't exist")
                continue
            load = [checkpoint, generation]
            break

    print('\n\nInitializing population\n')
    p = Population(n=20, name='demo', elitism_rate=0.25, min_species_size=5, monitor=Monitor(),
                   input_size=input_size, output_size=output_size, epochs=0,
                   train=functools.partial(
                       train_on_data,
                       torch_device=torch_device,
                       data_loader_train=data_loader_train
                   ),
                   evaluate=functools.partial(
                       evaluate,
                       torch_device=torch_device,
                       data_loader_test=data_loader_val,
                       output_size=output_size
                   ),
                   parent_selection=functools.partial(
                       stochastic_universal_sampling,
                       selection_percentage=0.3
                   ),
                   crossover=crossover,
                   load=load)
    for _ in range(50):
        p.evolve()


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
