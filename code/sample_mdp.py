import numpy as np

class sample_mdp:
    def __init__(self):
        self.state=1
    
    def take_action(self,move):
        reward=0
        terminal=False
        if self.state==1:
            if move == 0:
                reward=0.122
                self.state = np.random.choice([1,2],p=[0.66,0.34])
            elif move == 1:
                reward=0.033
                self.state = np.random.choice([1,2],p=[0.99,0.01])

        elif self.stat==2:
            reward=0
            terminal=True
        return self.state,reward,terminal
