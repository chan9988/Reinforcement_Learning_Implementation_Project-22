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


def train():

    ewma_rewaord=0

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

if __name__ == '__main__':
    random_seed = 22
    env = gym.make('LunarLander-v2')
    env.reset(seed=random_seed)  
    torch.manual_seed(random_seed) 
    train()
