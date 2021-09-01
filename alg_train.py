from alg_general_functions import *
from alg_logger import run
from alg_net import *
from alg_memory import *
from alg_module import *


def train():
    # Initialization

    # ENV
    env = gym.make(ENV)
    # state = env.reset()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # NETS
    critic_net = CriticNet(obs_size, n_actions)
    critic_target_net = CriticNet(obs_size, n_actions).load_state_dict(critic_net.state_dict())
    actor_net = ActorNet(obs_size, n_actions)
    actor_target_net = ActorNet(obs_size, n_actions).load_state_dict(actor_net.state_dict())

    # REPLAY BUFFER
    train_dataset = ALGDataset()
    fill_the_buffer(train_dataset, env, actor_net)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train
    DDPG_module = ALGModule()
    DDPG_module.train()

    # Save Results
    if SAVE_RESULTS:
        torch.save(actor_net, 'actor_net.pt')
        # example runs
        model = torch.load('actor_net.pt')
        model.eval()
        play(10, model=model)


if __name__ == '__main__':
    train()









