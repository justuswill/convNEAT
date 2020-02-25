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

from KMedoids import KMedoids
from genome import Genome
from net import build_net_from_genome


class Population:
    """
    A population of genomes to be evolved
    -----
    n                - population size
    evaluate         - how to get a score from genome
    parent_selection - how parents are selected from the population
    crossover        - how to combine genomes to form new ones
    train            - how to train net nets
    epochs           - the standard (minimum) number of epochs to train before evaluation
    min_species_size - the lower limit to species sizes (cur. only used for spawn calculations)
    elitism_rate     - the % of best genomes to transfer ot the new generation
    n_generations_no_change, tol         - how many generations a species is allowed to not improve by at least tol
    load = [checkpoint_name, generation] - whether to load from a checkpoint
    monitor          - if the results should be shown graphically
    """

    def __init__(self, n, evaluate, parent_selection, crossover, train, input_size, output_size,
                 name=None, elitism_rate=0.1, min_species_size=5, n_generations_no_change=5, tol=1e-5,
                 epochs=2, load=None, monitor=None):
        # Evolution parameters
        self.evaluate = evaluate
        self.parent_selection = parent_selection
        self.crossover = crossover
        self.train = train
        self.epochs = epochs
        self.min_species_size = min_species_size
        self.elitism_rate = elitism_rate
        self.n_generations_no_change = n_generations_no_change
        self.tol = tol

        # Plotting and tracking training progress
        self.monitor = monitor
        # On the go plotting of species score
        self.polygons = dict()

        # Calculated after first clustering
        self.species_repr = None

        # Load instead
        if load is not None:
            self.load_checkpoint(*load)
        else:
            # Historical markers starting at 5
            self.id_generator = itertools.count(5)
            self.species_id_generator = itertools.count(1)

            self.input_size = input_size
            self.output_size = output_size

            # Begin with only one species
            self.n = n
            self.number_of_species = 1
            self.species = {0: [Genome(self) for _ in range(n)]}
            self.best_genome = self.species[0][0].copy()
            self.top_score = 0
            self.history = []

            # Metadata
            self.generation = 1
            self.checkpoint_name = name or time.strftime("%d.%m-%H:%M")

    def next_id(self):
        return next(self.id_generator)

    def save_checkpoint(self):
        save = [self.n, self.id_generator, self.species_id_generator, self.number_of_species, self.generation,
                self.input_size, self.output_size, self.checkpoint_name, self.top_score, self.history,
                (self.best_genome.__class__, self.best_genome.save()),
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
            [self.n, self.id_generator, self.species_id_generator, self.number_of_species, self.generation,
             self.input_size, self.output_size, self.checkpoint_name, self.top_score, self.history,
             saved_best_genome, saved_genomes] = pickle.load(c)
            self.best_genome = saved_best_genome[0](self).load(saved_best_genome[1])
            self.species = {species: [genome[0](self).load(genome[1]) for genome in genomes]
                            for species, genomes in saved_genomes.items()}

    def cluster(self, threshold=120, rel_threshold=[1.2, 0.75]):
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
        k = self.number_of_species
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
        for i, sp in enumerate(sorted_species_ids):
            in_cluster_distances = distances[np.ix_(labels == sp, labels == sp)]
            cur_centers_by_species[sp] = np.argmin(np.sum(in_cluster_distances, axis=1)) + cumlen[i]
        cur_centers = [cur_centers_by_species[i] for i in sorted_species_ids]

        # Get performance of K-Medoids for some # of clusters near k
        ids_to_check = list(range(max(1, k - 2), min(int(n / self.min_species_size) + 1, k + 3)))
        medoids = {i: KMedoids(n_clusters=i, metric='precomputed', min_cluster_size=self.min_species_size).
                      fit(distances, old_centers=cur_centers)
                   for i in ids_to_check}
        all_labels = {i: medoid.labels_ for i, medoid in medoids.items()}
        scores = {i: medoid.score_ for i, medoid in medoids.items()}

        # Clustering performance
        logging.info("Scores for  clustering:\n" +
                     ", ".join(["%s: %s" % (sp, sc) for sp, sc in zip(ids_to_check, list(scores.values()))]))

        # Change number of clusters
        while k + 1 in ids_to_check and threshold * self.n <= scores[k + 1] < rel_threshold[1] * scores[k]:
            print("number of clusters increased by one")
            k += 1
            new_species = next(self.species_id_generator)
            sorted_species_ids += [new_species]
            species_ids += [new_species]
        while k - 1 in ids_to_check and (scores[k] < threshold * self.n or scores[k - 1] < rel_threshold[0] * scores[k]):
            print("number of clusters increased by one")
            k -= 1

        # Save new Clustering
        self.number_of_species = k
        # Use old identifiers for clusters
        labels = [sorted_species_ids[i] for i in all_labels[k]]
        self.species_repr = {sp: all_genomes[center]
                             for sp, center in zip(sorted_species_ids, medoids[k].medoid_indices_)}

        # Rebuild species
        self.species = {i: [] for i in species_ids if i in labels}
        for i, g in enumerate(all_genomes):
            self.species[labels[i]] += [g]

        # Save to history starting with newest
        self.history += [{species: [len(genomes), None]
                          for species, genomes in sorted(self.species.items(), key=lambda x: x[0], reverse=True)}]

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
            cieling = [[self.n, self.n]] if i > 0 else [[0, self.n]]
            lines = cieling + connections + new_sp + dead_sp
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

    def species_death(self, evaluated_genomes_by_species):
        """
        Check for species that haven't improved in <n_generations_no_change> and kill them.
        Elites are adopted by other species.
        The remaining space is filled by other species
        """
        if self.generation > self.n_generations_no_change:
            species_ids = list(evaluated_genomes_by_species.keys())
            past = [(sp, [self.history[sp][i]
                          for i in range(self.generation - self.n_generations_no_change, self.generation + 1)])
                    for sp in species_ids]
            for sp, past_scores in past:
                if max(past_scores) < past_scores[0] + self.tol:
                    # Get elites
                    elitism = math.ceil(self.elitism_rate * len(evaluated_genomes_by_species[sp]))
                    elite_genomes = [(g, s) for g, s in evaluated_genomes_by_species[sp][:elitism]]

                    # Decide who adopts them
                    while len(elite_genomes) > 0:
                        g, s = elite_genomes.pop(0)
                        new_sp = min(species_ids, key=lambda sp: g.dissimilarity(self.species_repr[sp]))
                        evaluated_genomes_by_species[new_sp] += [(g, s)]
                    del evaluated_genomes_by_species[sp]

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

        Updates plots for best and current net and fill scores of species
        Saves the net with its parameters for continuation of training later on (used by elites)
        Also saves weights in every gene as start for child genomes
        """
        counter = itertools.count(1)
        evaluated_genomes_by_species = dict()
        score_by_species = dict()
        for sp, genomes in sorted(self.species.items()):
            evaluated_genomes = []
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
                net, optim, criterion = build_net_from_genome(g, self.input_size, self.output_size)
                self.train(g, net, optim, criterion, epochs=self.epochs)
                score = self.evaluate(net)
                g.score = score

                # Show best net
                if score > self.top_score:
                    self.top_score = score
                    self.best_genome = g.copy()
                    if self.monitor is not None:
                        self.monitor.plot(0, (g.__class__, g.save(parameters=False)), kind='net-plot', title='best',
                                          input_size=(1, 28, 28), score=score, clear=True, show=True)

                evaluated_genomes += [(g, score)]
            evaluated_genomes_by_species[sp] = sorted(evaluated_genomes, key=lambda x: x[1], reverse=True)

            # Score of species is the mean of their genomes' scores
            sp_scores = [s for g, s in evaluated_genomes]
            score_by_species[sp] = sum(sp_scores) / len(sp_scores)

            # Update History
            self.history[-1][sp] = [self.history[-1][sp][0], score_by_species[sp]]

            # Fill species plot
            if self.monitor is not None:
                p = PatchCollection([self.polygons[sp]], cmap='viridis', alpha=0.4)
                colors = [score_by_species[sp]**3]
                p.set_array(np.array(colors))
                p.set_clim([0, 1])
                self.monitor.plot(2, p, kind='add_collection')
        return [evaluated_genomes_by_species, score_by_species]

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
                              input_size=self.input_size, score=self.top_score, title='best', clear=True, show=True)

        self.cluster()
        evaluated_genomes_by_species, score_by_species = self.train_nets()

        # Saving checkpoint with net parameters
        print("Saving checkpoint after training\n")
        self.save_checkpoint()

        print('\n\nGENERATION %d\n' % self.generation)
        for species, evaluated_genomes in evaluated_genomes_by_species.items():
            print('Species %d with %d members - mean acc %.2f:\n' %
                  (species, len(evaluated_genomes), score_by_species[species]))
            for g, s in evaluated_genomes:
                r = repr(g)
                if len(r) > 64:
                    r = r[:60] + '...' + r[-1:]
                print('%64s: %.4f' % (r, s))
            print()

        # Resize species, increase better scoring species and kill bad performing ones
        self.species_death(evaluated_genomes_by_species)
        new_sizes = self.new_species_sizes(score_by_species)

        print("Breading new neural networks")
        # Same mutation (split_edge) in a gen get the same innovation number
        this_gen_mutations = dict()
        for sp, evaluated_genomes in evaluated_genomes_by_species.items():
            old_n_sp = len(evaluated_genomes)
            new_n_sp = new_sizes[sp]
            elitism = min(math.ceil(self.elitism_rate * old_n_sp), new_n_sp)
            elite_genomes = [g for g, s in evaluated_genomes[:elitism]]
            print("%d Elites in species %d:" % (elitism, sp))
            for g in elite_genomes:
                print(g)
            print()
            parents = self.parent_selection(evaluated_genomes, k=new_n_sp-elitism)
            new_genomes = [self.crossover(p[0], p[1]).mutate_random(this_gen_mutations) for p in parents]
            self.species[sp] = elite_genomes + new_genomes

        self.generation += 1
