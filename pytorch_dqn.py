import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
# env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""

    def __init__(self,device):
        super(DQN, self).__init__()
        self.device=device
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, is_eval=False):
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learn_step_counter / EPS_DECAY)
        state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)  # get a 1D array
        action_value = self.eval_net.forward(state)
        action = torch.max(action_value, 1)[1].cpu().data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        if not is_eval and np.random.randn() < epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # update the parameters(directly from eval_net)
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int)).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:]).to(self.device)

        # q_eval
        # select  action but the action has the biggest prob(TD learning with target network)
        q_eval = self.eval_net(batch_state).gather(1, batch_action).to(self.device)
        # DQN:
        # q_next = self.target_net(batch_next_state).detach()
        # q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # Double DQN:
        indices = self.eval_net(batch_state).max(1)[1].unsqueeze(1).to(self.device)
        q_next = self.target_net(batch_next_state).gather(1, indices).to(self.device)
        q_target = batch_reward + GAMMA * q_next.view(BATCH_SIZE, 1).to(self.device)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def test_agent(dqn, test_episodes):
    test_reward_list=[]
    for i in range(test_episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state, is_eval=True)
            next_state, _, done, info = env.step(action)
            ep_reward += 1
            if done:
                break
            state = next_state
        print("test_episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
        test_reward_list.append(ep_reward)
    plt.plot(range(test_episodes), test_reward_list)
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {}".format(device))
    dqn = DQN(device)
    episodes = 220
    print("Collecting Experience....")
    reward_list = []
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, _, done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += 1

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        r = ep_reward
        reward_list.append(r)
    plt.plot(range(episodes), reward_list)
    plt.show()
    plt.close()
    test_agent(dqn, 30)

if __name__ == '__main__':
    main()
