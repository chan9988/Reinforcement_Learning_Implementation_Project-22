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



def select_action_Boltzmann_softmax(beta,action,c):
    p=[]
    for act in action:
        p.append( math.exp(beta * act - c) )
    sum_p = sum(p)
    for i in range(len(p)):
        p[i] = p[i]/sum_p

    return np.random.choice(range(0,len(p)),p=p)

def mm_omega(omega,action,c):
    m=[]
    for act in action:
        m.append( math.exp(omega * (act-c)) )
    mm = sum(m) / len(m)
    mm = c + math.log(mm) / omega
    return mm

def mm_function(beta,omega,action,c,mm):
    m=[]
    for act in action:
        m.append( math.exp(beta * (act-mm)) * (act-mm) )
    return sum(m)

def mm_calculate_beta(omega,action,c):
    mm = mm_omega(omega,action,c)
    sol = optimize.root_scalar(mm_function,args=(omega,action,c,mm), bracket=[-100, 100],method='brentq')
    return sol.root

def select_action_Mellowmax(omega,action,c):
    beta=mm_calculate_beta(omega,action,c)
    return select_action_Boltzmann_softmax(beta,action,c)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def forward(self, state):
        return

    def select_action(self, state):
        return

    def calculate_loss(self, gamma=0.99):
        return 

def train():

    model = Policy()
    omega = 5
    # for test
    action=[6,9.1,9,6]
    print(select_action_Boltzmann_softmax(omega,action,max(action)))
    print(select_action_Mellowmax(omega,action,max(action)))

    ewma_rewaord=0
'''
    for i_episode in itertools.count(1):
        state = env.reset()
    
        ep_reward=0
        max_episode_len = 10000
        for t in range(max_episode_len):
            action = np.random.choice([0,1,2,3])
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        
        print(ep_reward)
'''

if __name__ == '__main__':
    random_seed = 22
    env = gym.make('LunarLander-v2')
    env.reset(seed=random_seed)  
    torch.manual_seed(random_seed) 
    train()
