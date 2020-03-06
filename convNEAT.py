import functools
import random
import numpy as np
import logging
import os

import torch

from population import Population
from selection import cut_off_selection, tournament_selection, fitness_proportionate_selection,\
    fitness_proportionate_tournament_selection, linear_ranking_selection, stochastic_universal_sampling
from net import train_on_data, evaluate
from monitor import Monitor
from exploration import show_genomes, from_human_readable


def data_loader(data, batch_size=100, validation_size=0.15, **kwargs):
    """ Build data loaders """
    val = int(validation_size * len(data))
    data_train, data_val = torch.utils.data.random_split(data, [len(data) - val, val])
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)

    return data_loader_train, data_loader_val


class ConvNEAT:

    def __init__(self, output_size, n=100, torch_device='cpu', name=None, monitoring=True, seed=None, max_gens=50):
        # manually seed all random number generators for reproducible results
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)

            # Deterministic backend is slower, use only when needed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.n = n
        self.output_size = output_size
        self.torch_device = torch_device
        self.monitoring = monitoring
        self.name = name
        self.max_gens = max_gens

    def evolve(self, p):
        for i in range(self.max_gens):
            p.evolve()
            # If Converged
            if p.converged:
                print("Training successful")
                break
            if i == self.max_gens - 1:
                logging.warning("Further training could potentially increase performance.\n"
                                "Consider increasing max_generations for a better result.")

    def fit(self, data, load=None, **kwargs):

        input_size = list(data[0][0].shape)
        data_loader_train, data_loader_val = data_loader(data, **kwargs)

        print('\n\nInitializing population\n')
        p = Population(input_size=input_size, output_size=self.output_size, name=self.name, n=self.n,
                       monitor=Monitor() if self.monitoring else None, **kwargs,
                       train=functools.partial(
                           train_on_data,
                           torch_device=self.torch_device,
                           data_loader_train=data_loader_train
                       ),
                       evaluate=functools.partial(
                           evaluate,
                           torch_device=self.torch_device,
                           data_loader_test=data_loader_val,
                           output_size=self.output_size
                       ),
                       parent_selection=functools.partial(
                           stochastic_universal_sampling,
                           selection_percentage=0.3
                       ),
                       load=load)
        self.evolve(p)

    def prompt(self, data=None, **kwargs):
        """
        Prompt the user what to do.
        Choices are loading a checkpoint, exploring checkpoints, starting evolution, etc
        """

        input_size = list(data[0][0].shape)

        # Load from checkpoint?
        while True:
            loading = input("load from checkpoint? [y/n]")
            if loading == 'e':
                show_genomes(input_size=input_size)
                return
            if loading == 'f':
                data_loader_train, data_loader_val = data_loader(data, **kwargs)
                from_human_readable(input_size=input_size, output_size=self.output_size,
                                    evaluate=functools.partial(
                                        evaluate,
                                        torch_device=self.torch_device,
                                        data_loader_test=data_loader_val,
                                        output_size=self.output_size
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
        self.fit(data, load=load, **kwargs)
