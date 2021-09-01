from alg_constrants_amd_packages import *
from alg_logger import run

class ALGModule:
    def __init__(self):

        if PLOT_LIVE:
            self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    def train(self):
        pass
        # if NEPTUNE:
        #     run['acc_loss'].log(loss)
        #     run['acc_loss_log'].log(f'{loss}')
        #
        # self.plot({'rewards': rewards, 'values': values, 'ref_v': ref_v.numpy(),
        #            'loss': self.log_for_loss, 'lengths': lengths, 'adv_v': adv_v.numpy()})

    def plot(self, graph_dict):
        # plot live:
        if PLOT_LIVE:
            # plt.clf()
            # plt.plot(list(range(len(self.log_for_loss))), self.log_for_loss)
            # plt.plot(list(range(len(rewards))), rewards)

            ax = self.fig.get_axes()

            for indx, (k, v) in enumerate(graph_dict.items()):
                ax[indx].cla()
                ax[indx].plot(list(range(len(v))), v, c='r')  # , edgecolor='b')
                ax[indx].set_title(f'Plot: {k}')
                ax[indx].set_xlabel('iters')
                ax[indx].set_ylabel(f'{k}')

            plt.pause(0.05)
            # plt.pause(1.05)







