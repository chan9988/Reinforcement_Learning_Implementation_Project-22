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


def train():
    return


if __name__ == '__main__':
    random_seed = 22
    env = gym.make('LunarLanderContinuous-v2')
    env.reset(seed=random_seed)  
    torch.manual_seed(random_seed) 
    train()
