import itertools
import time
import os
import pickle
import math
import random
import numpy as np

from KMedoids import KMedoids
from genome import Genome


class Population:
    """
    A population of genomes to be evolved
    -----
    n                - population size
    evaluate_genome  - how to get a score from genome
    parent_selection - how parents are selected from the population
    elitism_rate     - the % of best genomes to transfer ot the new generation
    load = [checkpoint_name, generation] - whether to load from a checkpoint
    monitor          - if the results should be shown graphically
    """

    def __init__(self, n, evaluate_genome, parent_selection, crossover, name=None, elitism_rate=0.05,
                 min_species_size=4, load=None, monitor=None):
        # Evolution parameters
        self.n = n
        self.evaluate_genome = evaluate_genome
        self.parent_selection = parent_selection
        self.crossover = crossover
        self.elitism_rate = elitism_rate
        self.min_species_size = min_species_size

        # Load instead
        if load is not None:
            self.load_checkpoint(*load)
        self.monitor = monitor
        # For training progress
        self.i = itertools.count()

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
        ids_to_check = list(range(max(1, k - 2), min(int(n / self.min_species_size) + 1, k + 3)))
        all_labels = {i: KMedoids(n_clusters=i, metric='precomputed').fit(distances).labels_ for i in ids_to_check}
        scores = {i: self.cluster_score(distances, lab) for i, lab in all_labels.items()}

        # Plot clustering performance
        if self.monitor is not None:
            self.monitor.send([2, [ids_to_check, list(scores.values())], dict()])

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
        if self.monitor is not None:
            self.monitor.send([3, [distances[np.ix_(ind, ind)]], {'kind': 'imshow'}])

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

    def new_species_sizes(self, score_by_species):
        """
        Calculate new the new sizes for every species (proportionate to fitness)
        """
        scores = np.array(list(score_by_species.values()))
        sizes = np.maximum(scores/sum(scores) * self.n, self.min_species_size)
        sizes = np.around(sizes/sum(sizes) * self.n)

        # Force n genomes
        while sum(sizes) > self.n:
            print(sizes)
            r = random.choice(np.where(sizes > min_species_size))
            sizes[r] -= 1
        while sum(sizes) < self.n:
            print("2", sizes)
            sizes[random.choice(range(sizes.shape[0]))] += 1

        return {sp: int(size) for sp, size in zip(score_by_species.keys(), sizes)}

    def evolve(self):
        """
        Group the genomes to species and evaluate them on training data
        Generate the next generation with selection, crossover and mutation
        """
        # Saving checkpoint
        print("Saving checkpoint")
        self.save_checkpoint()

        self.cluster()
        # Reset training progress counter
        self.i = itertools.count()
        evaluated_genomes_by_species = {species: sorted([(g, self.evaluate_genome(g, monitor=self.monitor))
                                                         for g in genomes], key=lambda g: g[1], reverse=True)
                                        for species, genomes in self.species.items()}

        print('\n\nGENERATION %d\n' % self.generation)
        for species, evaluated_genomes in evaluated_genomes_by_species.items():
            print('Species %d with %d members:\n' % (species, len(evaluated_genomes)))
            for g, s in evaluated_genomes:
                r = repr(g)
                if len(r) > 64:
                    r = r[:60] + '...' + r[-1:]
                print('{:64}:'.format(r), s)
            print()

        # show best net
        if self.monitor is not None:
            best_of_species = [genomes[0] for genomes in evaluated_genomes_by_species.values()]
            best_genome, score = sorted(best_of_species, key=lambda x: x[1])[0]
            self.monitor.send([0, [(best_genome.__class__, best_genome.save())],
                               {'kind': 'net-plot', 'input_size': (1, 28, 28), 'score': score, 'title': 'best'}])

        # Score of species is mean of scores
        score_by_species = {species: sum([s for g, s in genomes]) / len(genomes)
                            for species, genomes in evaluated_genomes_by_species.items()}

        # Resize species, increase better scoring species
        new_sizes = self.new_species_sizes(score_by_species)

        print("Breading new neural networks")
        for sp, evaluated_genomes in evaluated_genomes_by_species.items():
            old_n_sp = len(evaluated_genomes)
            new_n_sp = new_sizes[sp]
            elitism = min(math.floor(self.elitism_rate * old_n_sp), new_n_sp)
            elite_genomes = [g for g, s in evaluated_genomes[:elitism]]
            parents = self.parent_selection(evaluated_genomes, k=new_n_sp-elitism)
            new_genomes = [self.crossover(p[0], p[1]) for p in parents]
            self.species[sp] = elite_genomes + new_genomes

        self.generation += 1
