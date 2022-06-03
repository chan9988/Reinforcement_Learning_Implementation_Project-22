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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mm_omega(omega,action,c):
    m=[]
    for act in action[0]:
        m.append( math.exp(omega * (act-c)) )
    mm = sum(m) / len(m)
    mm = c + math.log(mm) / omega
    return mm

def mm_function(beta,omega,action,c,mm):
    m=[]
    for act in action[0]:
        try:
            m.append( math.exp(beta * (act-mm)) * (act-mm) )
        except OverflowError:
            print(beta,act,mm)
        
    return sum(m)

def mm_calculate_beta(omega,action,c):
    mm = mm_omega(omega,action,c)
    sol = optimize.root_scalar(mm_function,args=(omega,action,c,mm), bracket=[-10, 10],method='brentq')
    return sol.root


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 
        self.hidden_size = 128

        self.layer1 = nn.Sequential(nn.Linear(self.observation_dim, self.hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_size, self.action_dim))

        self.saved_actions = []
        self.rewards = []  

    def select_action_Boltzmann_softmax(self,beta,action,c):
        action = beta * (action - c)
        action = F.softmax(action,dim=1)
        return action

    def select_action_Mellowmax(self,omega,action,c):
        beta = mm_calculate_beta(omega,action,c)
        #print(beta)
        return self.select_action_Boltzmann_softmax(beta,action,c)

    def forward(self, state):       
        x = self.layer1(state)
        x = self.layer2(x)
        #x = F.softmax(x,dim=1)
        #beta = 1.5
        #x = self.select_action_Boltzmann_softmax(1.5,x,max(x[0]).item())
        omega=1
        x = self.select_action_Mellowmax(omega,x,max(x[0]).item())

        return x

    def select_action(self, state):
        state=torch.tensor(state).float().unsqueeze(0).to(device)
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
            R = r*0.1 + gamma * R
            returns.insert(0,R) 
                  
        for L, R in zip(saved_actions, returns):
            policy_losses.append(- L * R)        
        
        loss=sum(policy_losses)
        loss.backward()                 
        
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]

def train():

    lr = 0.01

    # Instantiate the policy model and the optimizer
    model = Policy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    ewma_reward=0
    loss=0

    for i_episode in itertools.count(1):
        state = env.reset()
    
        ep_reward=0
        max_episode_len = 1000
        for t in range(max_episode_len):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        
        optimizer.zero_grad()
        loss=model.calculate_loss()
        optimizer.step()
        model.clear_memory()

        
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        
        
        if i_episode%10 == 0:
            print('Episode {}\tlength: {}\treward: {:.4f}\t ewma reward: {:.4f}\t loss: {}'.format(i_episode, t, ep_reward, ewma_reward, loss))
            
        if ewma_reward > env.spec.reward_threshold:
            #torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


if __name__ == '__main__':
    random_seed = 22
    env = gym.make('LunarLander-v2')
    env.reset(seed=random_seed)  
    torch.manual_seed(random_seed) 
    train()
