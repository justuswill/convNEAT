import os
import matplotlib.pyplot as plt

from population import Population


def show_genomes(evaluate):
    # What to show
    checkpoint = input("checkpoint name:")
    gens = input("generations ['all' / list separated by ' ']:")
    if gens == 'all':
        i = 0
        while os.path.exists(os.path.join("checkpoints", checkpoint, "%02d.cp" % i)):
            i += 1
        generations = list(range(i))
    else:
        generations = list(map(int, gens.split(" ")))

    for i in generations:
        p = Population(1, 1, 1, 1, load=(checkpoint, i))
        fig, axs = plt.subplots(len(p.species), 10)
        fig.canvas.set_window_title('Generation %d' % i)
        for j, sp in enumerate(sorted(p.species.keys(), key=lambda x: p.history[-1][x][1] or 0)):
            for k, g in enumerate(sorted(p.species[sp], key=lambda x: x.score or 0)):
                g.visualize(ax=axs[j, k], input_size=(1, 28, 28))
                axs[j, k].title.set_text('%sacc: %.2f %%' % ("Species %d " % sp if k == 0 else "", 100 * g.score))
    plt.show()
