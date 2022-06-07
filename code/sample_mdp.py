import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
from tqdm import tqdm


class sample_mdp:
    def __init__(self):
        self.state = 0

    def take_action(self, move):
        reward = 0
        terminal = False
        if self.state == 0:
            assert move in [0, 1]
            if move == 0:
                reward = 0.122
                self.state = np.random.choice([0, 1], p=[0.66, 0.34])
            elif move == 1:
                reward = 0.033
                self.state = np.random.choice([0, 1], p=[0.99, 0.01])

        if self.state == 1:
            terminal = True

        return self.state, reward, terminal


def select_action_Boltzmann_softmax(beta, Q_a, Q_b, c):
    p1 = math.exp(beta * Q_a - c) / (math.exp(beta * Q_a - c) + math.exp(beta * Q_b - c))
    p2 = math.exp(beta * Q_b - c) / (math.exp(beta * Q_a - c) + math.exp(beta * Q_b - c))
    return np.random.choice([0, 1], p=[p1, p2])


def mm_omega(omega, Q_a, Q_b, c):
    mm = (math.exp(omega * (Q_a - c)) + math.exp(omega * (Q_b - c))) / 2
    mm = c + math.log(mm) / omega
    return mm


def mm_function(beta, omega, Q_a, Q_b, c, mm):
    """Target function for solving beta for Maximum Entropy Mellomax Policy"""
    return math.exp(beta * (Q_a - mm)) * (Q_a - mm) + math.exp(beta * (Q_b - mm)) * (Q_b - mm)


def mm_calculate_beta(omega, Q_a, Q_b, c):
    mm = mm_omega(omega, Q_a, Q_b, c)
    sol = optimize.root_scalar(mm_function, args=(omega, Q_a, Q_b, c, mm), bracket=[-100, 100], method='brentq')
    return sol.root


def select_action_Mellowmax(omega, Q_a, Q_b, c):
    beta = mm_calculate_beta(omega, Q_a, Q_b, c)
    return select_action_Boltzmann_softmax(beta, Q_a, Q_b, c)


def segment_average(array: list, n_ele_per_seg: int = 10):
    n_smoothed_points = len(array) // 10 + (0 if len(array) % n_ele_per_seg == 0 else 1)
    segments = np.array_split(np.array(array), n_smoothed_points)
    new_array = []
    for seg in segments:
        new_array.append(np.mean(seg))
    return new_array


def train(method: str,
          Q_a=0,
          Q_b=0,
          episode=2000,
          alpha=0.1,
          gamma=0.98,
          beta=16.55,
          omega=16.55):
    assert method in ['Boltzmann Softmax', 'Mellowmax']

    x_episode = []
    y_qa = []
    y_qb = []

    for i_episode in tqdm(range(episode), desc='SARSA training'):
        env = sample_mdp()
        trajectory = [env.state]
        ter = False
        # print(i_episode)

        while not ter:
            # Original SARSA need to derive (r, s') and extract next a'
            # However, we don't need to compute it here since this toy problem has only one
            #     non-terminal state, i.e. s' must equal to s when it doesn't terminate.
            # If it's terminal, we don't need to compute Q(s',a') anymore and don't need a'.

            # action = np.random.choice([0,1],p=[0.5,0.5])
            if method == 'Boltzmann Softmax':
                action = select_action_Boltzmann_softmax(beta, Q_a, Q_b, max(Q_a, Q_b))
            elif method == 'Mellowmax':
                action = select_action_Mellowmax(omega, Q_a, Q_b, max(Q_a, Q_b))
            else:
                raise ValueError(f'Invalid method: "{method}".')

            state, reward, ter = env.take_action(action)

            if ter:
                if action == 0:
                    Q_a = Q_a + alpha * (reward - Q_a)
                elif action == 1:
                    Q_b = Q_b + alpha * (reward - Q_b)
            else:
                if action == 0:
                    Q_a = Q_a + alpha * (reward + gamma * Q_a - Q_a)
                elif action == 1:
                    Q_b = Q_b + alpha * (reward + gamma * Q_b - Q_b)

            trajectory.append(state)

        # print(Q_a,Q_b)
        x_episode.append(i_episode)
        y_qa.append(Q_a)
        y_qb.append(Q_b)

    plt.title(f'{method}')
    plt.xlabel('Episodes')
    plt.plot(x_episode, y_qa, label='^Q(s1,a)', color='green')
    plt.plot(x_episode, y_qb, label='^Q(s1,b)', color='blue')
    plt.legend()
    plt.show()

    # Smoothed by averaging over 10 consecutive points
    plt.title(f'{method} (smoothed)')
    plt.xlabel('Episodes')
    x_episode_smoothed = segment_average(x_episode, n_ele_per_seg=10)
    y_qa_smoothed = segment_average(y_qa, n_ele_per_seg=10)
    y_qb_smoothed = segment_average(y_qb, n_ele_per_seg=10)
    plt.plot(x_episode_smoothed, y_qa_smoothed, label='^Q(s1,a)', color='green')
    plt.plot(x_episode_smoothed, y_qb_smoothed, label='^Q(s1,b)', color='blue')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Method: ['Boltzmann Softmax', 'Mellowmax']
    train(method='Boltzmann Softmax')
    train(method='Mellowmax')

