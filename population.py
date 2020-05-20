import itertools
import time
import os
import pickle
import math
import random
import logging
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import torch

from KMedoids import KMedoids
from genome import Genome
from net import build_net_from_genome
from crossover import crossover
from tools import score_decay, check_cuda_memory


class Population:
    """
    A population of genomes to be evolved
    -----
    n                - population size
    evaluate         - how to get a acc from genome
    parent_selection - how parents are selected from the population
    crossover        - how to combine genomes to form new ones
    train            - how to train net nets
    epochs           - the standard (minimum) number of epochs to train before evaluation
    min_species_size - the lower limit to species sizes (cur. only used for spawn calculations)
    elitism_rate     - the % of best genomes to transfer ot the new generation
    n_generations_no_change, tol         - how many generations a species is allowed to not improve by at least tol
    load = [checkpoint_name, generation] - whether to load from a checkpoint
    save_mdoe        - all:     save all genome and gene parameters
                       elites:  save all elite  and gene parameters
                       genomes: save all genome          parameters
                       bare:    save all elite           parameters
                       None:    don't save               parameters
    monitor          - if the results should be shown graphically
    load_params      - if the weights etc should be loaded when using load
    """

    def __init__(self, n, input_size, output_size, evaluate, parent_selection, train, cross_over=crossover,
                 name=None, elitism_rate=0.1, min_species_size=5, n_generations_no_change=5, tol=1e-5,
                 mutate_speed=1, min_species=1, max_species=10, epochs=2, reward_epochs=10,
                 load=None, save_mode="elites", monitor=None, load_params=True):
        # Evolution parameters
        self.evaluate = evaluate
        self.parent_selection = parent_selection
        self.crossover = cross_over
        self.train = train
        self.epochs = epochs
        self.reward_epochs = reward_epochs
        self.min_species_size = min_species_size
        self.min_species = min_species
        self.max_species = max_species
        self.elitism_rate = elitism_rate
        self.mutate_speed = mutate_speed
        self.n_generations_no_change = n_generations_no_change
        self.tol = tol

        # Plotting and tracking training progress
        self.monitor = monitor
        self.polygons = dict()

        # Species centers calculated after first clustering
        self.species_repr = None
        self.converged = False

        # What to save: save_genomes =1 saves elites =2 saves all genomes
        self.save_genomes, self.save_genes = {"all": [2, True], "elites": [1, True], "genomes": [2, False],
                                              "bare": [1, False], "None": [0, False]}[save_mode]
        # Load if a checkpoint is given
        if load is not None:
            self.load_checkpoint(*load, load_params=load_params)
        else:
            # Historical markers starting at 5
            self.id_generator = itertools.count(5)
            self.species_id_generator = itertools.count(1)

            self.input_size = input_size
            self.output_size = output_size

            # Begin with min species of nearly same size
            self.n = n
            self.species = {i: [Genome(self) for _ in range(n // min_species if i > 0
                                                            else (n // min_species) + (n % min_species))]
                            for i in range(min_species)}
            self.best_genome = self.species[0][0].copy()
            self.top_acc = 0
            self.history = []

            # Metadata
            self.generation = 1
            self.checkpoint_name = name or time.strftime("%d.%m-%H:%M")

        self.check_args()

    def check_args(self):
        """ Checks if all parameters are correct """
        if self.min_species <= 0:
            raise ValueError("There always has to be a positive number of species")
        if self.min_species > self.max_species:
            raise ValueError("min_species (%d) has to be smaller than or equal max_species (%d)" %
                             (self.min_species, self.max_species))
        if self.min_species * self.min_species_size > self.n:
            raise ValueError("Can't achieve %d species with size %d.\n"
                             "Choose a higher n" % (self.min_species, self.min_species_size))

    def next_id(self):
        return next(self.id_generator)

    def save_checkpoint(self, update=False):
        if not update:
            # Remember the random state of the start or reproducibility
            self.this_gen_random_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())

        save = [self.n, self.id_generator, self.species_id_generator, self.generation,
                self.input_size, self.output_size, self.checkpoint_name, self.top_acc, self.history,
                self.this_gen_random_state,
                (self.best_genome.__class__, self.best_genome.save()),
                {species: [(genome.__class__, genome.save()) for genome in genomes]
                 for species, genomes in self.species.items()}]

        _dir = os.path.join('checkpoints', self.checkpoint_name)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open(os.path.join(_dir, "%02d.cp" % self.generation), "wb") as c:
            pickle.dump(save, c)

    def load_checkpoint(self, checkpoint_name, generation, load_params=True):
        file_path = os.path.join('checkpoints', checkpoint_name, "%02d.cp" % generation)
        with open(file_path, "rb") as c:
            [self.n, self.id_generator, self.species_id_generator, self.generation,
             self.input_size, self.output_size, self.checkpoint_name, self.top_acc, self.history,
             saved_random_state, saved_best_genome, saved_genomes] = pickle.load(c)
            random.setstate(saved_random_state[0])
            np.random.set_state(saved_random_state[1])
            torch.set_rng_state(saved_random_state[2])
            self.best_genome = saved_best_genome[0](self).load(saved_best_genome[1], load_params=load_params)
            self.species = {species: [genome[0](self).load(genome[1], load_params=load_params) for genome in genomes]
                            for species, genomes in saved_genomes.items()}

    def cluster(self, threshold=120, rel_threshold=(1.2, 0.85)):
        """
        Cluster the genomes with K_Medoids-Clustering
        Change the number of species if needed (-2 .. +2)

        The number of clusters is less than the minimum needed to achieve a score below <threshold>.
        i.e. the clustering always scores above <threshold> * n - with n = size of population
        This is to avoid small clusters, 1500 is good value for 50-100 genomes # TODO is it?

        Additionally,
        the number of cluster decreases if the score of k-1 is < <rel_threshold[0]> % of k score
        the number of cluster increases if the score of k+1 is < <rel_threshold[1]> % of k score
        """
        k = len(self.species)
        n = self.n

        # Sorted by size
        species_ids = list(self.species.keys())
        sorted_species_ids = sorted(species_ids, key=lambda i: len(self.species[i]))
        # Combine all species
        all_genomes = [g for i in species_ids for g in self.species[i]]

        # Distance matrix, species sorted by id
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = all_genomes[i].dissimilarity(all_genomes[j])

        # Get centers of old species, species sorted by size
        species_len = [len(self.species[sp]) for sp in species_ids]
        cumlen = np.cumsum([0] + species_len)
        cur_centers_by_species = dict()
        labels = np.array([sp for sp in self.species.keys() for _ in self.species[sp]])
        for i, sp in enumerate(species_ids):
            in_cluster_distances = distances[np.ix_(labels == sp, labels == sp)]
            cur_centers_by_species[sp] = np.argmin(np.sum(in_cluster_distances, axis=1)) + cumlen[i]
        cur_centers = [cur_centers_by_species[i] for i in sorted_species_ids]

        # Get performance of K-Medoids for some # of clusters near k
        low = max(self.min_species, k - 2)
        up = min(self.max_species, int(n / self.min_species_size), k + 2) + 1
        ids_to_check = list(range(low, up))
        medoids = {i: KMedoids(n_clusters=i, metric='precomputed', min_cluster_size=self.min_species_size).
                      fit(distances, old_centers=cur_centers)
                   for i in ids_to_check}
        all_labels = {i: medoid.labels_ for i, medoid in medoids.items()}
        scores = {i: medoid.score_ for i, medoid in medoids.items()}

        # Clustering performance
        logging.info("Scores for  clustering:\n" +
                     ", ".join(["%s: %s" % (sp, sc) for sp, sc in zip(ids_to_check, list(scores.values()))]))

        # Change number of clusters
        while k + 1 in ids_to_check and threshold * self.n <= scores[k+1] < rel_threshold[1] * scores[k]:
            print("number of clusters increased by one")
            k += 1
            new_species = next(self.species_id_generator)
            sorted_species_ids += [new_species]
            species_ids += [new_species]
        while k - 1 in ids_to_check and (scores[k] < threshold * self.n or scores[k-1] < rel_threshold[0] * scores[k]):
            print("number of clusters decreased by one")
            k -= 1

        # Use old identifiers for clusters
        labels = [sorted_species_ids[i] for i in all_labels[k]]
        self.species_repr = {sp: all_genomes[center]
                             for sp, center in zip(sorted_species_ids, medoids[k].medoid_indices_)}

        # Rebuild species
        self.species = {i: [] for i in species_ids if i in labels}
        for i, g in enumerate(all_genomes):
            self.species[labels[i]] += [g]

        # Save to history starting with newest
        entry = {species: [len(genomes), None]
                 for species, genomes in sorted(self.species.items(), key=lambda x: x[0], reverse=True)}
        if len(self.history) >= self.generation:
            # Compatible with laoded data
            self.history[self.generation - 1] = entry
        else:
            self.history += [entry]

        if self.monitor is not None:
            self.distance_plot(labels, distances)
            self.species_plot()

    def distance_plot(self, labels, distances):
        """
        Plot the pairwise distances between all genomes in the population
        Blue means similar, green/yellow means less similar
        Genomes are ordered by cluster and clusters are shown with red squares
        """
        species_len = np.cumsum([0] + [len(g) for g in self.species.values()])
        ind = np.argsort(labels)
        self.monitor.plot(3, distances[np.ix_(ind, ind)], kind='imshow', clear=True)
        for s, e in zip(species_len[:-1], species_len[1:]):
            self.monitor.plot(3, np.array([s, s, e, e, s]) - 0.5, np.array([s, e, e, s, s]) - 0.5,
                              c='red', linewidth=1.5)
        self.monitor.send()

    def species_plot(self):
        """
        Plot species sizes and their performance.
        x-axis shows the generations
        y-axis how big each species was in that generation
        the color of the shape from t-1 to t shows the score in Generation t
        """
        # Clear
        self.polygons = dict()
        self.monitor.plot(2, clear=True)

        scores = [np.array([sc for _, sc in hist.values()]) for hist in self.history[:-1]]
        lens = [[len_ for len_, _ in hist.values()] for hist in self.history]
        cumlens = [np.cumsum([0] + l) for l in lens]
        sp_ids = [hist.keys() for hist in self.history]
        pos = [{sp: pos for sp, pos in zip(sp_id, cumlen)} for sp_id, cumlen in zip(sp_ids, cumlens)]
        # Special start
        pos = [{0: 0}] + pos
        sp_ids = [pos[0].keys()] + sp_ids
        cumlens = [[0]] + cumlens

        # For every generation
        for i in range(len(pos) - 1):
            # survived species
            connections = [[pos[i][sp], pos[i + 1][sp]] for sp in sp_ids[i] & sp_ids[i + 1]]
            # new species
            new_sp = [[pos[i][max([p for p in sp_ids[i] if p < sp])] if min(sp_ids[i]) < sp else max(cumlens[i]),
                       pos[i + 1][sp]]
                      for sp in sp_ids[i + 1] - sp_ids[i]]
            # killed species
            dead_sp = [[pos[i][sp], pos[i + 1][max([p for p in sp_ids[i + 1] if p < sp])]
                       if min(sp_ids[i + 1]) < sp else self.n]
                       for sp in sp_ids[i] - sp_ids[i + 1]]
            # Line on top
            ceiling = [[self.n, self.n]] if i > 0 else [[0, self.n]]
            lines = ceiling + connections + new_sp + dead_sp
            # Plot scores
            patches = []
            for sp in sp_ids[i + 1]:
                base = [sp if sp in sp_ids[i] else max([p for p in sp_ids[i] if p < sp])
                        if min(sp_ids[i]) < sp else self.n if i > 0 else 0, sp]
                bel = [pos[i][base[0]], pos[i + 1][base[1]]]
                abv = list(map(lambda x: pos[x][max([p for p in sp_ids[x] if p < sp])]
                               if min(sp_ids[x]) < sp else self.n if x > 0 else 0, [i, i + 1]))
                polygon = Polygon([[i, bel[0]], [i, abv[0]], [i + 1, abv[1]], [i + 1, bel[1]]], True)
                patches.append(polygon)
                # If no score yet, fill later
                if i == len(scores):
                    self.polygons.update({sp: polygon})
            # If score, plot now
            if i < len(scores):
                p = PatchCollection(patches, cmap='viridis', alpha=0.4)
                colors = scores[i] ** 3
                p.set_array(np.array(colors))
                p.set_clim([0, 1])
                self.monitor.plot(2, p, kind='add_collection')
            self.monitor.plot(2, [i, i + 1], [[l for l, _ in lines], [l for _, l in lines]], c='darkblue')
            self.monitor.plot(2, list(range(self.generation + 1)), kind="set_xticks")
        self.monitor.send()

    def species_death(self, evaluated_genomes_by_species, score_by_species):
        """
        Check for species that haven't improved their mean accuracy in <n_generations_no_change> and kill them.
        Elites are adopted by other species.
        The remaining space is filled by other species

        Returns changed score_by_species
        """
        killed = False
        if self.generation > self.n_generations_no_change:
            species_ids = list(evaluated_genomes_by_species.keys())
            start = self.generation - self.n_generations_no_change - 1
            past = {sp: [self.history[i][sp][1] for i in range(start, self.generation)]
                    for sp in species_ids if sp in self.history[start].keys()}
            logging.info("Species past: %s" % past)
            for sp, past_scores in past.items():
                if len(species_ids) <= self.min_species:
                    break
                if max(past_scores[1:]) < past_scores[0] + self.tol:
                    killed = True

                    # Get elites
                    elitism = math.ceil(self.elitism_rate * len(evaluated_genomes_by_species[sp]))
                    elite_genomes = [(g, s) for g, s in evaluated_genomes_by_species[sp][:elitism]]

                    # Decide who adopts them
                    while len(elite_genomes) > 0:
                        g, s = elite_genomes.pop(0)
                        new_sp = min(species_ids, key=lambda sp: g.dissimilarity(self.species_repr[sp]))
                        evaluated_genomes_by_species[new_sp] += [(g, s)]

                    # Delete genomes
                    del evaluated_genomes_by_species[sp]
                    species_ids.remove(sp)

                    print("Species %d killed due to low performance" % sp)
        if not killed:
            return score_by_species
        else:
            # Recompute score_by_species
            scores_by_species = {sp: [s for g, s in evaluated_genomes_by_species[sp]] for sp in species_ids}
            return {sp: sum(scores_by_species[sp]) / len(scores_by_species[sp]) for sp in species_ids}

    def new_species_sizes(self, score_by_species):
        """
        Calculate the new sizes for every species (proportionate to fitness)
        """
        scores = np.array(list(score_by_species.values()))
        sizes = np.maximum(scores/sum(scores) * self.n, self.min_species_size)
        sizes = np.around(sizes/sum(sizes) * self.n)

        # Force n genomes
        while sum(sizes) > self.n:
            r = random.choice(list(np.where(sizes > self.min_species_size)))
            sizes[r] -= 1
        while sum(sizes) < self.n:
            sizes[random.choice(range(sizes.shape[0]))] += 1

        return {sp: int(size) for sp, size in zip(score_by_species.keys(), sizes)}

    def train_nets(self):
        """
        Train a instantiated net for every genome in the population
        Evaluate every net and return the scored genomes and species

        Updates plots for best and current net and fill scores/acc of species
        Saves the net with its parameters for continuation of training later on (used by elites)
        Also saves weights in every gene as start for child genomes
        """
        counter = itertools.count(1)
        evaluated_genomes_by_species = dict()
        score_by_species = dict()
        acc_by_species = dict()
        for sp, genomes in sorted(self.species.items()):
            evaluated_genomes = []
            sp_scores = []
            sp_accs = []
            for g in genomes:
                i = next(counter)
                print('Instantiating neural network from the following genome in species %d - (%d/%d):' %
                      (sp, i, self.n))
                print(g)

                # Visualize current net
                if self.monitor is not None:
                    self.monitor.plot(1, (g.__class__, g.save(parameters=False)), kind='net-plot', title='train',
                                      n=self.n, i=i, input_size=self.input_size, clear=True, show=True)

                logging.debug('Building Net')
                try:
                    net, optim, criterion = build_net_from_genome(g, self.input_size, self.output_size)
                    logging.info("Cuda Usage %d - before training" % len(check_cuda_memory()))
                    self.train(g, net, optim, criterion, epochs=self.epochs + g.reward,
                               save_net_param=self.save_genomes >= 1, save_gene_param=self.save_genes)
                    g.reward = 0
                    logging.info("Cuda Usage %d - after training" % len(check_cuda_memory()))
                    acc = self.evaluate(net)
                    logging.info("Cuda Usage %d - after evaluation" % len(check_cuda_memory()))
                except RuntimeError as e:
                    logging.info("Net failed to train:\n%s" % e)
                    acc = 0
                g.acc = acc
                score = score_decay(acc, g.trained)

                # Show best net
                if acc > self.top_acc:
                    self.top_acc = acc
                    self.best_genome = g.copy()
                    if self.monitor is not None:
                        self.monitor.plot(0, (g.__class__, g.save(parameters=False)), kind='net-plot', title='best',
                                          input_size=(1, 28, 28), acc=acc, clear=True, show=True)

                evaluated_genomes += [(g, score)]
                sp_scores += [score]
                sp_accs += [acc]
            evaluated_genomes_by_species[sp] = sorted(evaluated_genomes, key=lambda x: x[1], reverse=True)

            # Score/acc of species is the mean of their genomes' scores
            score_by_species[sp] = sum(sp_scores) / len(sp_scores)
            acc_by_species[sp] = sum(sp_accs) / len(sp_accs)

            # Update History with acc
            self.history[-1][sp] = [self.history[-1][sp][0], acc_by_species[sp]]

            # Fill species plot
            if self.monitor is not None:
                p = PatchCollection([self.polygons[sp]], cmap='viridis', alpha=0.4)
                colors = [score_by_species[sp]**3]
                p.set_array(np.array(colors))
                p.set_clim([0, 1])
                self.monitor.plot(2, p, kind='add_collection')
        return [evaluated_genomes_by_species, score_by_species, acc_by_species]

    def rewards(self, evaluated_genomes_by_species, score_by_species):
        """
        The best performing nets get extra time to train so that faster progress can be made
        Epochs are awarded proportionate to species score and ranked to genomes
        """
        # Can be rounded to < reward_epochs
        rewards_by_species = {sp: int(self.reward_epochs * score/sum(score_by_species.values()))
                              for sp, score in score_by_species.items()}
        for sp, reward in rewards_by_species.items():
            for g, s in evaluated_genomes_by_species[sp]:
                g.reward = reward // 2
                reward -= reward // 2
                if reward <= 0:
                    break

    def evolve(self):
        """
        Group the genomes to species and evaluate them on training data
        Generate the next generation with selection, crossover and mutation
        """
        # Saving checkpoint
        print("Saving checkpoint\n")
        self.save_checkpoint()

        # show best net
        if self.monitor is not None:
            self.monitor.plot(0, (self.best_genome.__class__, self.best_genome.save(parameters=False)), kind='net-plot',
                              input_size=self.input_size, acc=self.top_acc, title='best', clear=True, show=True)

        self.cluster()
        evaluated_genomes_by_species, score_by_species, acc_by_species = self.train_nets()

        # Saving checkpoint with net parameters
        print("Saving checkpoint after training\n")
        self.save_checkpoint(update=True)

        print('\n\nGENERATION %d\n' % self.generation)
        for species, evaluated_genomes in evaluated_genomes_by_species.items():
            print('Species %d with %d members - score: %.2f, mean acc %.2f:\n' %
                  (species, len(evaluated_genomes), score_by_species[species], acc_by_species[species]))
            for g, s in evaluated_genomes:
                r = repr(g)
                if len(r) > 64:
                    r = r[:60] + '...' + r[-1:]
                print('%64s: %.4f' % (r, s))
            print()

        # Resize species, increase better scoring species and kill bad performing ones
        score_by_species = self.species_death(evaluated_genomes_by_species, score_by_species)
        new_sizes = self.new_species_sizes(score_by_species)

        self.rewards(evaluated_genomes_by_species, score_by_species)

        print("Breading new neural networks")
        # Same mutations (split_edge) in a gen get the same innovation number
        this_gen_mutations = dict()
        self.species = dict()
        for sp, evaluated_genomes in evaluated_genomes_by_species.items():
            # Save elites
            old_n_sp = len(evaluated_genomes)
            new_n_sp = new_sizes[sp]
            elitism = min(math.ceil(self.elitism_rate * old_n_sp), new_n_sp)
            elite_genomes = [g for g, s in evaluated_genomes[:elitism]]
            print("%d Elites in species %d:" % (elitism, sp))
            for g in elite_genomes:
                print(g)
            print()

            # Delete net parameters of non-elites
            if self.save_genes < 2:
                for g, _ in evaluated_genomes[elitism:]:
                    g.net_parameters = None

            # Selection & Crossover & Mutation
            parents = self.parent_selection(evaluated_genomes, k=new_n_sp-elitism)
            new_genomes = [self.crossover(p[0], p[1]).mutate_random(this_gen_mutations, exception=self.mutate_speed)
                           for p in parents]
            self.species[sp] = elite_genomes + new_genomes

        x = len([g for sp, genomes in self.species.items() for g in genomes])
        if x != self.n:
            print("Error")
            print()
            logging.error("Error occured in evolution step")

        self.generation += 1
