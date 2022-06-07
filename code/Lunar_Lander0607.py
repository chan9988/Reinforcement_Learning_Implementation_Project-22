import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
import itertools
from time import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############## changed ###################
def mm_omega(omega, action, c):
    mm = torch.exp(omega*(action[0]-c)).mean()
    mm = c+torch.log(mm)/omega
    return mm


def mm_function(beta, action, mm):
    try:
        m = (torch.exp(beta*(action[0]-mm))*(action[0]-mm)).sum()
    except OverflowError:
        print(beta, action, mm)
    return m
############## ####### ###################


def mm_calculate_beta(omega, action, c):
    mm = mm_omega(omega, action, c)
    sol = optimize.root_scalar(mm_function, args=(
        action, mm), bracket=[-10, 10], method='brentq')
    return sol.root


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size = 128

        self.layer1 = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim))
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        # modify here to use Boltzmann or Mellowmax
        omega = 5
        beta = 3  # Boltzmann
        #beta = mm_calculate_beta(omega, x, x.max().item())  # Mellowmax

        ############## changed ###################
        # action = beta * (action - c)
        x = beta * x
        ############## ####### ###################
        x = F.softmax(x, dim=1)

        return x

    def select_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(m.log_prob(action))

        return action.item()

    def calculate_loss(self, gamma=0.99):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        returns = []

        for r in self.rewards[::-1]:
            ############## changed ###################
            # R = r*0.1 + gamma * R
            R = r + gamma * R
            ############## ####### ###################
            returns.insert(0, R)

        for L, R in zip(saved_actions, returns):
            policy_losses.append(- L * R)

        loss = sum(policy_losses)
        loss.backward()

        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]


# plot the training curve
# name_ = f'mello_5_0.005'
name_ = f'boltz_3_0.005'
print('='*30)
print(name_)
print('='*30)


def plot_graph(ewmas):
    plt.title(name_)
    plt.xlabel('episode')
    plt.ylabel('ewma reward')
    plt.plot(ewmas)
    plt.savefig(f'training_curve_{name_}.png')
    plt.close()


def train():

    # lr = 0.01
    lr = 0.005

    # Instantiate the policy model and the optimizer
    model = Policy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    ewma_reward_list = []
    ewma_reward = 0
    best_ewma_reward = 0
    loss = 0

    for i_episode in itertools.count(1):
        state = env.reset()

        ep_reward = 0
        max_episode_len = 1000
        s = time()
        for t in range(max_episode_len):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        e = time()
        optimizer.zero_grad()
        loss = model.calculate_loss()
        model.clear_memory()
        optimizer.step()
        scheduler.step()
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        ewma_reward_list.append(ewma_reward)
        if i_episode % 20 == 0:
            plot_graph(ewma_reward_list)
            print('Episode {}, length: {}, reward: {:.4f}, ewma reward: {:.4f}, loss: {:.4f}, time: {:.1f}'
                  .format(i_episode, t, ep_reward, ewma_reward, loss.item(), e-s))
        if best_ewma_reward < ewma_reward:
            best_ewma_reward = ewma_reward
            torch.save(model.state_dict(),
                       f'./preTrained/lunar_{name_}_best.pth')
        if ewma_reward > env.spec.reward_threshold:
            plot_graph(ewma_reward_list)
            torch.save(model, f'lunar_{name_}_last.pt')
            torch.save(ewma_reward_list, f'ewma/ewma_{name_}.pt')
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


if __name__ == '__main__':
    random_seed = 22
    env = gym.make('LunarLander-v2')
    env.reset(seed=random_seed)
    torch.manual_seed(random_seed)
    train()
