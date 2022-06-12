from cmath import inf
import numpy as np
import copy
import statistics
import math
from scipy import optimize

class taxi_domain:
    def __init__(self):
        '''
          i
        j   0 1 2 3 4 5 6
          0 S X F O X O D
          1 O X O O X O O
          2 O O O O O O O
          3 X X O O O X X
          4 O O O O O O F
          5 F O O O O O X
            
          pos = i + 7*j
        '''
        self.pos=0
        self.f1=0 # 0 2
        self.f2=0 # 4 6
        self.f3=0 # 5 0
        '''
            state = pos*8 + f1*4 + f2*2 + f3 
            pos have 33 possible values
            total have 33*2*2*2 = 264 states
        '''
        self.state= self.pos*8 + self.f1 * 4 + self.f2 * 2 + self.f3
        self.reward=0
        self.terminal=False
    
        self.cnt=100000 # if cnt<0 terminal = True
        
    
    def move(self,action):
        self.cnt-=1
        '''
            action: 0:left 1:up 2:right 3:down
        '''
        if action == 0:
            if self.pos == 0 or self.pos == 2 or self.pos == 5 or self.pos == 7 or self.pos == 9 or self.pos == 12 or self.pos == 14 or self.pos == 23 or self.pos == 28 or self.pos == 35:
                self.pos = self.pos
            else:
                self.pos -= 1

        elif action == 1:
            if self.pos == 0 or self.pos == 2 or self.pos == 3 or self.pos == 5 or self.pos == 6 or self.pos == 15 or self.pos == 18 or self.pos == 28 or self.pos == 29 or self.pos == 33 or self.pos == 34:
                self.pos = self.pos
            else:
                self.pos -= 7

        elif action == 2:
            if self.pos == 0 or self.pos == 3 or self.pos == 6 or self.pos == 7 or self.pos == 10 or self.pos == 13 or self.pos == 20 or self.pos == 25 or self.pos == 34 or self.pos == 40:
                self.pos = self.pos
            else:
                self.pos += 1

        elif action == 3:
            if self.pos == 14 or self.pos == 15 or self.pos == 19 or self.pos == 20 or self.pos == 34 or self.pos == 35 or self.pos == 36 or self.pos == 37 or self.pos == 38 or self.pos == 39 or self.pos == 40:
                self.pos = self.pos
            else:
                self.pos += 7
        
        if self.pos == 6:
            self.terminal=True
        if self.pos == 2:
            self.f1 = 1
        if self.pos == 34:
            self.f2 = 1
        if self.pos == 35:
            self.f3 = 1

        self.state= self.pos*8 + self.f1 * 4 + self.f2 * 2 + self.f3

        if self.terminal:
            sum=self.f1+self.f2+self.f3
            if sum == 0:
                self.reward = 0
            elif sum == 1:
                self.reward = 1
            elif sum == 2:
                self.reward = 3
            elif sum == 3:
                self.reward = 15
            return self.state,self.reward,self.terminal
        else:
            if self.cnt<=0:
                self.terminal=True
                return self.state,0,self.terminal
            else:
                return self.state,0,self.terminal

def Q_value_init():
    #Q_value=np.zeros(336*4)
    Q_value=np.random.rand(336*4)
    # reset => (pos,action)
    reset=[[0,0],[2,0],[5,0],[7,0],[9,0],[12,0],[14,0],[23,0],[28,0],[35,0],
            [0,1],[2,1],[3,1],[5,1],[6,1],[15,1],[18,1],[28,1],[29,1],[33,1],[34,1],
            [0,2],[3,2],[6,2],[7,2],[10,2],[13,2],[20,2],[25,2],[34,2],[40,2],
            [14,3],[15,3],[19,3],[20,3],[34,3],[35,3],[36,3],[37,3],[38,3],[39,3],[40,3]]
    for ind in reset:
        pos=ind[0]
        act=ind[1]
        for f in range(8):
            Q_value[ ((pos*8) + f) * 4 + act] = -inf

    return Q_value

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
        try:
            m.append( math.exp(beta * (act-mm)) * (act-mm) )
        except OverflowError:
            print(beta,act,mm)
        
    return sum(m)

def mm_calculate_beta(omega,action,c):
    mm = mm_omega(omega,action,c)
    sol = optimize.root_scalar(mm_function,args=(omega,action,c,mm), bracket=[-10, 10],method='brentq')
    return sol.root

def train(method,Q_value,epsilon=0.1,beta=1,omega=10,alpha=0.1,gamma=0.99):
    if not (method == "epi-greedy" or method == "boltz" or method == "mm"):
        print("wrong method")
        return

    cnt=100000

    while cnt>=0:
        cnt-=1

        env=taxi_domain()
        state=env.state # s
        state1=0 # s'
        ter=env.terminal
        action=0 # a
        action1=0 # a'
        reward=0
        
        trajectory=[]
        trajectory.append(state/8)

        # select action a
        if method == "epi-greedy":
            exploration = np.random.choice([True,False],p=[epsilon,1-epsilon])
            if exploration:
                action = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
                while Q_value[state*4+action] == -inf:
                    action = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
            else:
                p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
                action = list(p).index(max(p))

        elif method == "boltz":
            p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
            for i in range(len(p)):
                p[i] = math.exp(beta*p[i])
            s=sum(p)
            for i in range(len(p)):
                p[i] = p[i]/s

            action = np.random.choice([0,1,2,3],p=p)
            while Q_value[state*4+action] == -inf:
                action = np.random.choice([0,1,2,3],p=p)
            
        elif method == "mm":
            p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
            beta=mm_calculate_beta(omega,p,max(p))
            
            for i in range(len(p)):
                p[i] = math.exp(beta*p[i])
            s=sum(p)
            for i in range(len(p)):
                p[i] = p[i]/s

            action = np.random.choice([0,1,2,3],p=p)
            while Q_value[state*4+action] == -inf:
                action = np.random.choice([0,1,2,3],p=p)

        while not ter:
            state1,reward,ter = env.move(action) # take action a get s'
            trajectory.append(state1/8)

            # select action a'
            if method == "epi-greedy":
                exploration = np.random.choice([True,False],p=[epsilon,1-epsilon])
                if exploration:
                    action1 = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
                    while Q_value[state1 * 4 + action1] == -inf:
                        action1 = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
                else:
                    p=[Q_value[state1*4+0],Q_value[state1*4+1],Q_value[state1*4+2],Q_value[state1*4+3]]
                    action1 = list(p).index(max(p))

            elif method == "boltz":
                p=[Q_value[state1*4+0],Q_value[state1*4+1],Q_value[state1*4+2],Q_value[state1*4+3]]
                for i in range(len(p)):
                    p[i] = math.exp(beta*p[i])
                s=sum(p)
                for i in range(len(p)):
                    p[i] = p[i]/s
                
                action1 = np.random.choice([0,1,2,3],p=p)
                while Q_value[state1*4+action1] == -inf:
                    action1 = np.random.choice([0,1,2,3],p=p)

            elif method == "mm":
                p=[Q_value[state1*4+0],Q_value[state1*4+1],Q_value[state1*4+2],Q_value[state1*4+3]]
                beta=mm_calculate_beta(omega,p,max(p))
                for i in range(len(p)):
                    p[i] = math.exp(beta*p[i])
                s=sum(p)
                for i in range(len(p)):
                    p[i] = p[i]/s
                
                action1 = np.random.choice([0,1,2,3],p=p)
                while Q_value[state1*4+action1] == -inf:
                    action1 = np.random.choice([0,1,2,3],p=p)

            # update
            Q_value[state*4+action] = Q_value[state*4+action] + alpha * ( reward + gamma * Q_value[ state1*4 + action1] - Q_value[ state*4 + action] ) 
            # s <- s', a <- a'
            state=state1
            action=action1
        
        print(cnt,len(trajectory),reward)
        
        
    return Q_value

def test(method,Q_value,epsilon=0.1,beta=1,omega=10,alpha=0.1,gamma=0.99,round=100):
    if not (method == "epi-greedy" or method == "boltz" or method == "mm"):
        print("wrong method")
        return
    
    while round >0:
        round -= 1

        env=taxi_domain()
        state=env.state # s
        state1=0 # s'
        ter=env.terminal
        action=0 # a
        reward=0

        trajectory=[]
        trajectory.append(state/8)

        while not ter:
            # select action a

            if method == "epi-greedy":
                exploration = np.random.choice([True,False],p=[epsilon,1-epsilon])
                if exploration:
                    action = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
                    while Q_value[state*4+action] == -inf:
                        action = np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
                else:
                    p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
                    action = list(p).index(max(p))

            elif method == "boltz":
                p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
                for i in range(len(p)):
                    p[i] = math.exp(beta*p[i])
                s=sum(p)
                for i in range(len(p)):
                    p[i] = p[i]/s

                action = np.random.choice([0,1,2,3],p=p)
                while Q_value[state*4+action] == -inf:
                    action = np.random.choice([0,1,2,3],p=p)
                
            elif method == "mm":
                p=[Q_value[state*4+0],Q_value[state*4+1],Q_value[state*4+2],Q_value[state*4+3]]
                beta=mm_calculate_beta(omega,p,max(p))
                
                for i in range(len(p)):
                    p[i] = math.exp(beta*p[i])
                s=sum(p)
                for i in range(len(p)):
                    p[i] = p[i]/s

                action = np.random.choice([0,1,2,3],p=p)
                while Q_value[state*4+action] == -inf:
                    action = np.random.choice([0,1,2,3],p=p)
            
            state1,reward,ter = env.move(action) # take action a get s'
            trajectory.append(state1/8)

            state=state1
        
        print(trajectory)
        print(reward)
          


#Q_value=Q_value_init()

#Q_value = np.loadtxt('Q_epi-greedy.txt')
#Q_value = train("epi-greedy", Q_value, epsilon=0.1)
#test("epi-greedy", Q_value, epsilon=0.1)
#np.savetxt('Q_epi-greedy.txt', Q_value)

#Q_value = np.loadtxt('Q_boltz.txt')
#Q_value = train("boltz", Q_value, beta=1)
#test("boltz", Q_value, beta=1)
#np.savetxt('Q_boltz.txt', Q_value)

#Q_value = np.loadtxt('Q_mm.txt')
#Q_value = train("mm", Q_value, omega=1)
#test("mm", Q_value, omega=1)
#np.savetxt('Q_mm.txt', Q_value)
