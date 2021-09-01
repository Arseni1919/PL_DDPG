import gym
import torch.utils.data

from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_memory import ALGDataset
from alg_logger import run


class ALGModule:
    def __init__(
            self,
            env: gym.Env,
            critic_net: nn.Module,
            critic_target_net: nn.Module,
            actor_net: nn.Module,
            actor_target_net: nn.Module,
            train_dataset: ALGDataset,
            train_dataloader: torch.utils.data.DataLoader
    ):
        self.env = env
        self.critic_net = critic_net
        self.critic_target_net = critic_target_net
        self.actor_net = actor_net
        self.actor_target_net = actor_target_net
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

        if PLOT_LIVE:
            self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    def fit(self):
        state = self.env.reset()

        for step in range(MAX_STEPS):

            self.validation_step(step)

            action = get_action(state, self.actor_net)
            next_state, reward, done, _ = self.env.step(action)

            experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
            self.train_dataset.append(experience)

            state = next_state if done else self.env.reset()

            self.training_step(step)

        self.env.close()



    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        pass

    def training_step(self, step):
        if step % UPDATE_EVERY == 0:
            for batch in self.train_dataloader:
                states, actions, rewards, dones, new_states = batch
                # compute targets
                # TODO

                # update critic - gradient descent
                # TODO

                # update actor - gradient ascent
                # TODO

                # update target networks
                # TODO

                self.neptune_update(loss=None)
                self.plot(
                    {}
                    # {'rewards': rewards, 'values': values, 'ref_v': ref_v.numpy(), 'loss': self.log_for_loss, 'lengths': lengths, 'adv_v': adv_v.numpy()}
                )

    def validation_step(self, step):
        if step % VAL_CHECKPOINT_INTERVAL == 0:
            print(f'[VALIDATION STEP] Step: {step}')
            play(1, self.actor_target_net)

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

    @staticmethod
    def neptune_update(loss):
        if NEPTUNE:
            run['acc_loss'].log(loss)
            run['acc_loss_log'].log(f'{loss}')







