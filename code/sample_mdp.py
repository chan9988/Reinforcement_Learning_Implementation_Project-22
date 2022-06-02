import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize

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

def select_action_Boltzmann_softmax(beta,Q_a,Q_b,c):
    p1= math.exp(beta * Q_a - c)/(math.exp(beta * Q_a - c)+math.exp(beta * Q_b - c))
    p2= math.exp(beta * Q_b - c)/(math.exp(beta * Q_a - c)+math.exp(beta * Q_b - c))
    return np.random.choice([0,1],p=[p1,p2])

def mm_omega(omega,Q_a,Q_b,c):
    mm = (math.exp(omega * (Q_a-c))+math.exp(omega * (Q_b-c)) ) / 2
    mm = c + math.log(mm) / omega
    return mm

def mm_function(beta,omega,Q_a,Q_b,c,mm):
    return math.exp(beta * (Q_a-mm)) * (Q_a-mm) + math.exp(beta * (Q_b-mm)) * (Q_b-mm)

def mm_calculate_beta(omega,Q_a,Q_b,c):
    mm = mm_omega(omega,Q_a,Q_b,c)
    sol = optimize.root_scalar(mm_function,args=(omega,Q_a,Q_b,c,mm), bracket=[-100, 100],method='brentq')
    return sol.root

def select_action_Mellowmax(omega,Q_a,Q_b,c):
    beta=mm_calculate_beta(omega,Q_a,Q_b,c)
    return select_action_Boltzmann_softmax(beta,Q_a,Q_b,c)

def train():
    Q_a=0
    Q_b=0
    episode=20000
    alpha=0.1
    gamma=0.98
    beta=16.55
    omega=16.55

    x_episode=[]
    y_qa=[]
    y_qb=[]

    for i_episode in range(episode):
        env=sample_mdp()
        trajectory=[]
        trajectory.append(env.state)
        ter=False
        #print(i_episode)

        while not ter:
            #action = np.random.choice([0,1],p=[0.5,0.5])
            #action = select_action_Boltzmann_softmax(beta,Q_a,Q_b,max(Q_a,Q_b))
            action = select_action_Mellowmax(omega,Q_a,Q_b,max(Q_a,Q_b))
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
        
        #print(Q_a,Q_b)
        x_episode.append(i_episode)
        y_qa.append(Q_a)
        y_qb.append(Q_b)

    plt.plot(x_episode,y_qa,x_episode,y_qb)
    plt.show()

if __name__ == '__main__':
    train()