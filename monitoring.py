import matplotlib.pyplot as plt


def monitoring(conn):
    """
    To be executed in an other Process, allowing asynchronous plotting and
    examination of data while the training can continue
    """
    plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.canvas.set_window_title('convNeat')
    ax[0, 0].title.set_text('Best performing Net:')
    ax[0, 1].title.set_text('Currently training:')
    ax[1, 0].title.set_text('Cluster-Performance')
    ax[1, 1].title.set_text('Distance matrix')
    while True:
        if conn.poll():
            recv = conn.recv()
            cax = ax[recv[0] // 2, recv[0] % 2]
            # clear except title
            [line.remove() for line in cax.lines + cax.collections + cax.texts]
            args = recv[1]
            kwargs = recv[2]
            if 'kind' in kwargs.keys():
                # visualize graph
                if kwargs['kind'] == 'net-plot':
                    str_genome = args[0]
                    genome = str_genome[0](0).load(str_genome[1])
                    genome.visualize(cax, input_size=kwargs['input_size'])
                    if kwargs['title'] == 'train':
                        cax.title.set_text('Currently training (%d/%d):' % (kwargs['i'], kwargs['n']))
                    if kwargs['title'] == 'best':
                        cax.title.set_text('Best net - acc: %.2f' % kwargs['score'])
                # Other pyplot function
                else:
                    getattr(cax, kwargs['kind'])(*args, **{k: v for k, v in kwargs.items() if k != 'kind'})
            # ax.plot
            else:
                cax.plot(*args, **kwargs)
            # Take time to let __main__ process reach next monitor call
            plt.pause(1)
        plt.pause(5)
