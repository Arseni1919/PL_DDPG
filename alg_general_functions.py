from alg_constrants_amd_packages import *


def fill_the_buffer(train_dataset, env):
    state = env.reset()
    while len(train_dataset) < REPLAY_BUFFER_SIZE:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # env.render()
        if done:
            experience = Experience(state=state, action=action, reward=reward, done=done, new_state=None)
            state = env.reset()
        else:
            experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
        train_dataset.append(experience)
    env.close()


def get_action(state, net):
    _, policy_dist = net(np.expand_dims(state, axis=0))
    # _, action = torch.max(policy_dist.squeeze(), dim=1)
    # return int(action.item())
    action = torch.argmax(policy_dist.squeeze()).item()
    # print(action)
    return action


def play(times: int = 1, model=None):
    env = gym.make(ENV)
    state = env.reset()

    # model = ALGNet(env.observation_space.shape[0], env.action_space.n)
    # model.load_state_dict(torch.load("example.ckpt"))

    game = 0
    total_reward = 0
    while game < times:
        action = env.action_space.sample()
        # action = get_action(state, model)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            state = env.reset()
            game += 1
            print(f'finished game {game} with a total reward: {total_reward}')
            total_reward = 0
        else:
            state = next_state
    env.close()