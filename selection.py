import numpy as np
import random
import math


# These selections all assume evaluated_genomes to be a sorted list of genomes
# where the first in the list is the fittest

def cut_off_selection(evaluated_genomes, k, survival_threshold=0.4):
    """
    Cut off population at a threshold. At least two parents survive.
    Couples are randomly sampled after the cut-off
    """
    n = len(evaluated_genomes)
    m = max(2, math.floor(survival_threshold * n))
    indices = [random.sample(range(m), k=2) for _ in range(k)]
    indices = [sorted(ind, key=lambda x: evaluated_genomes[x][1], reverse=True) for ind in indices]
    return [list(map(lambda i: evaluated_genomes[ind[i]][0], [0, 1])) for ind in indices]


def tournament_selection(evaluated_genomes, k, tournament_size=4):
    """
    Choose the two best individuals in a random sample of the population with set size
    """
    n = len(evaluated_genomes)
    tournament_size = max(2, min(n, tournament_size))
    tournaments = [random.sample(range(n), k=tournament_size) for _ in range(k)]
    return [list(map(lambda i: evaluated_genomes[sorted(t)[i]][0], [0, 1])) for t in tournaments]


def fitness_proportionate_selection(evaluated_genomes, k):
    """
    Sample random couples weighted by their score
    """
    n = len(evaluated_genomes)
    scores = [s for g, s in evaluated_genomes]
    scores = np.array(scores) / sum(scores)
    indices = [list(np.random.choice(range(n), size=2, p=scores, replace=False)) for _ in range(k)]
    indices = [sorted(ind, key=lambda x: evaluated_genomes[x][1], reverse=True) for ind in indices]
    return [list(map(lambda i: evaluated_genomes[ind[i]][0], [0, 1])) for ind in indices]


def linear_ranking_selection(evaluated_genomes, k):
    """
    Only the rank determines how the sampling is weighted.
    Chance of selection for the k-th of n individuals is 2k/(n(n-1))
    """
    n = len(evaluated_genomes)
    ranks = np.flip(np.arange(1, n+1))
    indices = [list(np.random.choice(range(n), size=2, p=ranks/sum(ranks), replace=False)) for _ in range(k)]
    indices = [sorted(ind, key=lambda x: evaluated_genomes[x][1], reverse=True) for ind in indices]
    return [list(map(lambda i: evaluated_genomes[ind[i]][0], [0, 1])) for ind in indices]


def fitness_proportionate_tournament_selection(evaluated_genomes, k, tournament_size=3):
    """
    Tournament selection but sampling is weighted by the score
    """
    n = len(evaluated_genomes)
    tournament_size = max(2, min(n, tournament_size))
    indices = [np.random.choice(range(n), size=tournament_size, replace=False) for _ in range(k)]
    indices = [sorted(ind, key=lambda x: evaluated_genomes[x][1], reverse=True) for ind in indices]
    return [list(map(lambda i: evaluated_genomes[ind[i]][0], [0, 1])) for ind in indices]


def stochastic_universal_sampling(evaluated_genomes, k, selection_percentage=0.3):
    """
    Select via SUS and than sample random couples.
    Does not avoid identical parents
    """
    n = len(evaluated_genomes)
    m = max(2, math.floor(selection_percentage * n))
    scores = [s for g, s in evaluated_genomes]
    step = sum(scores)/m
    start = random.uniform(0, step)
    cum_scores = np.cumsum(scores)
    parents = [evaluated_genomes[np.where(cum_scores > start + i*step)[0][0]][0] for i in range(m)]
    indices = [random.sample(range(m), k=2) for _ in range(k)]
    return [list(map(lambda i: parents[ind[i]], [0, 1])) for ind in indices]


if __name__ == '__main__':
    evaluated_genomes = [2*[n] for n in range(10)]

    for f in [cut_off_selection, tournament_selection, fitness_proportionate_selection,
              fitness_proportionate_tournament_selection, linear_ranking_selection, stochastic_universal_sampling]:
        out = ""
        for n in [2, 5, 10, 20]:
            k = math.floor(0.95 * n)
            l = [[i, i**(3/2)] for i in range(n, 0, -1)]
            scores = {i: 0 for i in range(1, n+1)}
            for j in range(5000):
                p = f(l, k=k)
                for c in p:
                    for t in c:
                        scores[t] += 1
            pro = np.array(list(scores.values()))
            pro = pro/sum(pro)
            for pp in pro:
                out += " %.2f" % pp
            out += " |"
        print(out)


