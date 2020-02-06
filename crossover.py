import random

from genome import Genome


def crossover(genome1, genome2, more_fit_crossover_rate=0.8, less_fit_crossover_rate=0.2):
    """
    Input:
    genome1: the more fit genome
    genome2: the less fit genome
    more_fit_crossover_rate: the rate at which genes only occurring in the more fit gene are used
    less_fit_crossover_rate: -"-
    -----
    Combine the genome to get a child-genome.
    Mutate the child genome.
    """
    population = genome1.population
    child_genes = []
    child_nodes = []

    disabled_ids = []

    # Genes
    ids_1, ids_2 = map(lambda x: set(x.genes_by_id.keys()), [genome1, genome2])
    for _id in ids_1 | ids_2:
        if _id in ids_1:
            if _id in ids_2:
                child_genes += [genome1.genes_by_id[_id].copy()]
            else:
                gene = genome1.genes_by_id[_id].copy()
                if random.random() > more_fit_crossover_rate:
                    disabled_ids += [_id]
                child_genes += [gene]
        else:
            gene = genome2.genes_by_id[_id].copy()
            if random.random() > less_fit_crossover_rate:
                disabled_ids += [_id]
            child_genes += [gene]

    # Nodes
    node_ids_1, node_ids_2 = map(lambda x: set(x.nodes_by_id.keys()), [genome1, genome2])
    for _id in node_ids_1 | node_ids_2:
        if _id in node_ids_1:
            child_nodes += [genome1.nodes_by_id[_id].copy()]
        else:
            child_nodes += [genome2.nodes_by_id[_id].copy()]

    child_genome = Genome(population, optimizer=genome1.optimizer,
                          nodes_and_genes=[child_nodes, child_genes])

    random.shuffle(disabled_ids)
    for _id in disabled_ids:
        child_genome.disable_edge(child_genome.genes_by_id[_id])

    return child_genome
