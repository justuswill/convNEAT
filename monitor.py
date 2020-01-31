import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe


class Monitor:
    """
    Plotting To be executed in an other Process, allowing asynchronous plotting and
    examination of data while the training can continue
    """

    def __init__(self):
        # Start Plot and connections
        child_conn, self.conn = Pipe()
        process = Process(target=self.monitoring, args=(child_conn,))
        process.start()

    def plot(self, ax_id, *args, **kwargs):
        """
        Send instructions to the other process
        """
        self.conn.send([ax_id, args, kwargs])

    @staticmethod
    def monitoring(conn, update=3):
        """
        Update plot every <update> seconds
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.ion()
        ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        fig.canvas.set_window_title('convNeat')
        ax[0].title.set_text('Best performing Net:')
        ax[1].title.set_text('Currently training:')
        ax[2].title.set_text('Cluster-Performance')
        ax[3].title.set_text('Distance matrix')

        while True:
            # if there is something new
            if conn.poll():
                recv = conn.recv()
                cax = ax[recv[0]]
                args = recv[1]
                kwargs = recv[2]

                # clear except title
                add = kwargs.pop('add', False)
                if not add:
                    title = cax.title
                    cax.clear()
                    cax.title = title
                if 'kind' in kwargs.keys():
                    kind = kwargs.pop('kind')
                    # visualize graph
                    if kind == 'net-plot':
                        str_genome = args[0]
                        genome = str_genome[0](0).load(str_genome[1])
                        genome.visualize(cax, input_size=kwargs['input_size'])
                        if kwargs['title'] == 'train':
                            cax.title.set_text('Currently training (%d/%d):' % (kwargs['i'], kwargs['n']))
                        if kwargs['title'] == 'best':
                            cax.title.set_text('Best net - acc: %.2f' % kwargs['score'])
                    # Other pyplot function
                    else:
                        getattr(cax, kind)(*args, **kwargs)
                # ax.plot
                else:
                    cax.plot(*args, **kwargs)
                # Take time to let __main__ process reach next monitor call
                plt.pause(0.1)
            plt.pause(update)
