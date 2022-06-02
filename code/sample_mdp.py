import numpy as np
import matplotlib.pyplot as plt

class sample_mdp:
    def __init__(self):
        self.state=0
    
    def take_action(self,move):
        reward=0
        terminal=False
        if self.state==0:
            if move == 0:
                reward=0.122
                self.state = np.random.choice([0,1],p=[0.66,0.34])
            elif move == 1:
                reward=0.033
                self.state = np.random.choice([0,1],p=[0.99,0.01])

        if self.state==1:
            terminal=True

        return self.state,reward,terminal



def train():
    Q_a=0
    Q_b=0
    episode=1000
    alpha=0.1
    gamma=0.98
    for _ in range(episode):
        env=sample_mdp()
        trajectory=[]
        trajectory.append(env.state)
        ter=False
        while not ter:
            action = np.random.choice([0,1],p=[0.5,0.5])
            state,reward,ter=env.take_action(action)
            
            if ter:
                if action == 0:
                    Q_a = Q_a+alpha * (reward - Q_a)
                elif action == 1:
                    Q_b = Q_b+alpha * (reward - Q_b)
            else:
                if action == 0:
                    Q_a = Q_a+alpha * (reward + gamma * Q_a - Q_a)
                elif action == 1:
                    Q_b = Q_b+alpha * (reward + gamma * Q_b - Q_b)

            trajectory.append(state)
        
        print(Q_a,Q_b)

train()