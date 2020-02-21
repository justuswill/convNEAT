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
        self.queue = []

    def send(self):
        """
        Send all queued plot tasks to the other process
        """
        self.conn.send(self.queue)
        self.queue = []

    def plot(self, ax_id, *args, **kwargs):
        """
        Queue a plotting instruction
        """
        show = kwargs.pop('show', False)
        self.queue.append([ax_id, args, kwargs])
        if show:
            self.send()

    @staticmethod
    def monitoring(conn, update=0.5):
        """
        Update plot every <update> seconds
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.ion()
        ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        fig.canvas.set_window_title('convNeat')
        ax[0].title.set_text('Best performing Net:')
        ax[1].title.set_text('Currently training:')
        ax[2].title.set_text('Species-Performance')
        ax[2].set_xlabel("Generation")
        ax[2].set_ylabel("# of genomes")
        ax[3].title.set_text('Distance matrix')
        plt.show()

        while True:
            # if there is something new
            if conn.poll():
                recv = conn.recv()
                for ax_id, args, kwargs in recv:
                    cax = ax[ax_id]
                    # clear except title/labels
                    clear = kwargs.pop('clear', False)
                    if clear:
                        title, xlabel, ylabel = [cax.title, cax.get_xlabel(), cax.get_ylabel()]
                        cax.clear()
                        cax.title = title
                        cax.set_xlabel(xlabel)
                        cax.set_ylabel(ylabel)
                    if 'kind' in kwargs.keys():
                        kind = kwargs.pop('kind')
                        # visualize graph
                        if kind == 'net-plot':
                            str_genome = args[0]
                            genome = str_genome[0](0).load(str_genome[1])
                            title = kwargs.pop('title')
                            if title == 'train':
                                cax.title.set_text('Currently training (%d/%d):' % (kwargs.pop('i'), kwargs.pop('n')))
                            elif title == 'best':
                                cax.title.set_text('Best net - acc: %.2f %%' % (100 * kwargs.pop('score')))
                            genome.visualize(cax, **kwargs)
                        # Other pyplot function
                        else:
                            getattr(cax, kind)(*args, **kwargs)
                    # ax.plot
                    elif len(args) > 0:
                        cax.plot(*args, **kwargs)
            fig.canvas.start_event_loop(update)
