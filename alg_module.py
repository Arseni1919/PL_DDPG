import gym
import torch.utils.data

from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_memory import ALGDataset
from alg_net import ActorNet, CriticNet
from alg_logger import run


class ALGModule:
    def __init__(
            self,
            env: gym.Env,
            critic_net: CriticNet,
            critic_target_net: CriticNet,
            actor_net: ActorNet,
            actor_target_net: ActorNet,
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

        self.critic_opt, self.actor_opt = self.configure_optimizers()

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

    def training_step(self, step):
        if step % UPDATE_EVERY == 0:
            print(f'[TRAINING STEP] Step: {step}')

            for batch in self.train_dataloader:
                states, actions, rewards, dones, next_states = batch

                # compute targets
                actions_target_net = self.actor_target_net(next_states)
                Q_target_vals = self.critic_target_net(next_states, actions_target_net)
                y = rewards + GAMMA * (1 - dones) * Q_target_vals

                # update critic - gradient descent
                self.critic_opt.zero_grad()
                actions_net = self.actor_net(states)
                Q_vals = self.critic_net(states, actions_net)
                critic_loss = nn.MSELoss(Q_vals, y)
                critic_loss.backward()
                self.critic_opt.step()

                # update actor - gradient ascent
                self.actor_opt.zero_grad()
                actions_net = self.actor_net(states)
                actor_loss = self.critic_net(states, actions_net).mean()
                actor_loss.backward()
                self.actor_opt.step()

                # update target networks
                for target_param, param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
                    target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

                for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
                    target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

                self.neptune_update(loss=None)
                self.plot(
                    {}
                    # {'rewards': rewards, 'values': values, 'ref_v': ref_v.numpy(), 'loss': self.log_for_loss, 'lengths': lengths, 'adv_v': adv_v.numpy()}
                )

    def validation_step(self, step):
        if step % VAL_CHECKPOINT_INTERVAL == 0:
            print(f'[VALIDATION STEP] Step: {step}')
            play(1, self.actor_target_net)

    def configure_optimizers(self):
        critic_opt = torch.optim.Adam(self.critic_net.parameters(), lr=LR)
        actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=LR)
        return critic_opt, actor_opt

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







