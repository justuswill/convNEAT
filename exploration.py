import os
import matplotlib.pyplot as plt

import torch

from population import Population
from genome import Genome
from optimizer import ADAMGene, SGDGene
from node import Node
from gene import KernelGene, PoolGene, DenseGene
from net import build_net_from_genome


def decode(line):
    """
    Reconstructs genome from str printed to console
    """
    alias = {"depth_change": "depth_size_change", "str": "stride", "pad": "padding", "d_mult": "depth_mult",
             "pool": "pooling", "mom": "momentum",
             "size": lambda x: [["width", "height"], x.split("*")] if not x == "None" else [[], []],
             "ID": lambda x: [["id"], [int(x)]] if "+" not in x else
                             [["id", "id_in", "id_out"], list(map(int, x.split("+")))]}
    # Stack of objects to be build. Form [class, attributes, Subobject in creation]
    object_stack = []
    for part in line.replace("|", ",").replace("->", "+").replace("(", "+").replace(")", ",").replace(">", ",>").\
            replace("[", "[,").replace("]", ",]").replace("\n", "").replace("False", "enabled=False").\
            replace("True", "enabled=True").split(","):
        if "<" in part or "[" in part:
            # Start genome
            if len(object_stack) == 0:
                object_stack.append([Genome, {"population": 1}, None])
            # New object attribute
            else:
                if "=" not in part:
                    name = None
                    strobj = part
                else:
                    name, strobj = part.split("=")
                obj = None
                for o, n in [(ADAMGene, "ADAMGene"), (SGDGene, "SGDGene"), (KernelGene, "KernelGene"),
                             (PoolGene, "PoolGene"), (DenseGene, "DenseGene"), (Node, "Node"), (list, "[")]:
                    if n in strobj:
                        obj = o
                        break
                if name is not None:
                    object_stack[-1][2] = name.replace(" ", "")
                # Add to Object Stack
                if obj == list:
                    object_stack.append([obj, [], None])
                else:
                    object_stack.append([obj, dict(), None])
        # List done
        elif part == "]":
            obj_raw = object_stack.pop(-1)
            object_stack[-1][1].update({object_stack[-1][2]: obj_raw[1]})
        # Object done
        elif part == ">":
            obj_raw = object_stack.pop(-1)
            obj = obj_raw[0](**obj_raw[1])
            if len(object_stack) == 0:
                # Genome constructed
                obj.nodes_by_id[0].role = 'input'
                obj.nodes_by_id[1].role = 'flatten'
                obj.nodes_by_id[2].role = 'output'
                return obj
            # Add the object attribute
            if isinstance(object_stack[-1][1], list):
                object_stack[-1][1] += [obj]
            else:
                object_stack[-1][1].update({object_stack[-1][2]: obj})
        elif "=" in part:
            # kwarg
            k, v = part.replace(" ", "").split("=")
            if callable(alias.get(k, k)):
                ks, vs = alias.get(k)(v)
            else:
                ks, vs = [alias.get(k, k)], [v]
            for i, v in enumerate(vs):
                try:
                    vs[i] = int(v)
                except (ValueError, TypeError):
                    try:
                        vs[i] = float(v)
                    except (ValueError, TypeError):
                        pass
            for k, v in zip(ks, vs):
                object_stack[-1][1].update({k: v})
        else:
            # arg (only in lists)
            v = part.replace(" ", "")
            try:
                v = int(v)
            except ValueError:
                pass
            object_stack[-1][1] += [v]


def from_human_readable(evaluate, input_size, output_size):
    while True:
        file = input("check genomes from file:")
        if not os.path.exists(file):
            continue
        with open(file, 'r') as f:
            for line in f.readlines():
                if "Genome" in line:
                    genome = decode(line)
                    fig, ax = plt.subplots()
                    net, _, _ = build_net_from_genome(genome, input_size, output_size)
                    genome.visualize(ax=ax, input_size=(1, 28, 28))
                    evaluate(net)
                    plt.show()
        return


def show_genomes(input_size, simultan=False):
    # What to show
    checkpoint = input("checkpoint name:")
    gens = input("generations ['all' / list separated by ' ']:")
    if gens == 'all':
        i = 1
        while os.path.exists(os.path.join("checkpoints", checkpoint, "%02d.cp" % i)):
            i += 1
        generations = list(range(1, i))
    else:
        generations = list(map(int, gens.split(" ")))

    for i in generations:
        p = Population(1, 1, 1, 1, 1, 1, 1, load=(checkpoint, i), load_params=False)
        show = 7
        fig, axs = plt.subplots(len(p.species), show, squeeze=False, figsize=(20, 10))
        fig.canvas.set_window_title('Generation %d' % i)
        for j, sp in enumerate(sorted(p.species.keys(), key=lambda x: p.history[-1][x][1] or 0, reverse=True)):
            for k, g in enumerate(sorted(p.species[sp], key=lambda x: x.acc or 0, reverse=True)[:show]):
                g.visualize(ax=axs[j, k], input_size=input_size)
                axs[j, k].title.set_text('%sacc: %s %%' % ("Species %d " % sp if k == 0 else "",
                                                           "%.2f" % (100 * g.acc) if g.acc is not None else "-"))
        if simultan:
            plt.pause(0.01)
        else:
            plt.show()
    plt.show()


"""
    # Plot Species representative ?
    for i in generations:
        p = Population(1, 1, 1, 1, 1, 1, 1, load=(checkpoint, i))
        show = 7
        fig, axs = plt.subplots(len(p.species), show, squeeze=False, figsize=(20, 10))
        fig.canvas.set_window_title('Generation %d' % i)
        for j, sp in enumerate(sorted(p.species.keys(), key=lambda x: p.history[-1][x][1] or 0, reverse=True)):
            p.species_repr[sp].visualize(ax=axs[j, 0], input_size=input_size)
            axs[j, k].title.set_text('species %d, mean_acc: %s %%' % (sp, p.history[j-1][sp][1] * 100))
            for k, g in enumerate(sorted(p.species[sp], key=lambda x: x.score or 0, reverse=True)[:show]):
                g.visualize(ax=axs[j, k], input_size=input_size)
                axs[j, k].title.set_text('acc: %.2f %%' % ((100 * g.score) if g.score is not None else "-"))
        plt.pause(0.01)
"""