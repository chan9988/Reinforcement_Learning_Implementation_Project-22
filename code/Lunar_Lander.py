import gym
import matplotlib.pyplot as plt
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


def mm_calculate_beta(omega, action, c):
    mm = mm_omega(omega, action, c)
    sol = optimize.root_scalar(mm_function, args=(
        action, mm), bracket=[-10, 10], method='brentq')
    return sol.root


b = 11
mello = True


class Policy(nn.Module):
    def __init__(self, mello):
        super(Policy, self).__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size = 16

        self.layer1 = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim))
        self.saved_actions = []
        self.rewards = []
        self.mello = mello
        self.omega = b

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        if self.mello:
            beta = mm_calculate_beta(
                self.omega, x, x.max().item())  # Mellowmax
        else:
            beta = b  # Boltzmann
        x = beta * x
        x = F.softmax(x, dim=1)

        return x

    def select_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(m.log_prob(action))

        return action.item()

    def calculate_loss(self, batch, gamma=0.99):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        for L, R in zip(saved_actions, returns):
            policy_losses.append(- L * R)

        loss = sum(policy_losses)/batch

        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]


def plot_graph(ewmas, means, exp_name=f'boltz_3_0.005'):
    # plot the training curve
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(exp_name)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('ewma reward')
    ax1.plot(ewmas)

    ax2.set_title(exp_name)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('mean returns')
    ax2.plot(means)
    plt.savefig(f'curves/{exp_name}.png')
    plt.close()
    torch.save(ewmas, f'ewma/ewma_{exp_name}.pt')
    torch.save(means, f'mean/mean_{exp_name}.pt')


def train(exp_name=f'boltz_3_0.005', mello=True):

    lr = 0.005

    # Instantiate the policy model and the optimizer
    model = Policy(mello).to(device)
    # model.load_state_dict(torch.load('preTrained/lunar_boltz_2_h16_best.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ewma_reward_list = []
    mean_return_list = []
    ewma_reward = 0
    mean_return = 0
    best_ewma_reward = 0
    batch_loss = 0
    batch = 10
    s = time()
    for i_episode in itertools.count(1):
        state = env.reset()

        ep_reward = 0
        max_episode_len = 1000

        for t in range(max_episode_len):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # loss = model.calculate_loss()
        batch_loss = batch_loss + model.calculate_loss(batch)
        if i_episode % batch == 0:
            # for loss in batch_loss:
            batch_loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
        model.clear_memory()
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        mean_return = mean_return+(ep_reward-mean_return)/i_episode
        mean_return_list.append(mean_return)
        ewma_reward_list.append(ewma_reward)
        if i_episode % 100 == 0:
            e = time()
            plot_graph(ewma_reward_list, mean_return_list, exp_name)
            print('Episode {}, length: {}, reward: {:.4f}, ewma reward: {:.4f}, loss: {:.4f}, time: {:.1f}'
                  .format(i_episode, t, ep_reward, ewma_reward, batch_loss.item(), e-s))
            s = time()
        if i_episode % batch == 0:
            batch_loss = 0
        if best_ewma_reward < ewma_reward:
            best_ewma_reward = ewma_reward
            torch.save(model.state_dict(),
                       f'./preTrained/lunar_{exp_name}_best.pth')
        # if ewma_reward > env.spec.reward_threshold:
        #     plot_graph(ewma_reward_list)
        #     torch.save(model, f'last/lunar_{exp_name}_last.pt')
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(ewma_reward, t))
        #     print(f'Total episodes: {i_episode}')
        #     break
        if i_episode >= 15000:
            plot_graph(ewma_reward_list, mean_return_list)
            if ewma_reward > env.spec.reward_threshold:
                torch.save(model.state_dict(),
                           f'last/lunar_{exp_name}_last_YES.pth')
            else:
                torch.save(model.state_dict(),
                           f'last/lunar_{exp_name}_last_NO.pth')
            print("Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            print(f'Total episodes: {i_episode}')
            break


if __name__ == '__main__':
    random_seed = 22
    for random_seed in [22, 7654, 321]:
        env = gym.make('LunarLander-v2')
        env.reset(seed=random_seed)
        torch.manual_seed(random_seed)
        if mello:
            exp_name = f'mello_{b}_b10_seed{random_seed}'
        else:
            exp_name = f'boltz_{b}_b10_seed{random_seed}'

        print('='*30)
        print(exp_name)
        print('='*30)
        train(exp_name, mello)
