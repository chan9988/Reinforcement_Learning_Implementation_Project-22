"""Random MDPs
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import glob
import os
from PIL import Image


# noinspection DuplicatedCode
def construct_mdp():
    # Construct MDPs
    # Paper settings:
    # n_states = np.random.choice(np.arange(2, 10 + 1, 1), size=None, replace=True, p=None)
    # n_actions = np.random.choice(np.arange(2, 5 + 1, 1), size=None, replace=True, p=None)
    n_states = np.random.choice(np.arange(2, 5 + 1, 1), size=None, replace=True, p=None)
    n_actions = np.random.choice(np.arange(2, 4 + 1, 1), size=None, replace=True, p=None)

    P_mat = np.random.random((n_states, n_actions, n_states)) / 100.  # It's [0,1), a little diff from paper's [0,1]
    # To be more similar to figure 1 in paper, consider R(s,a) here instead of R(s,a,s')
    R_mat = np.random.random((n_states, n_actions)) / 100.  # It's [0,1), a little diff from paper's [0,1]

    # P_mat
    for s in range(n_states):
        for a in range(n_actions):
            for s_ in range(n_states):
                P_mat[s, a, s_] += np.random.choice([np.random.normal(loc=1., scale=np.sqrt(0.1)), 0.], p=[0.5, 0.5])
                P_mat[s, a, s_] += np.random.choice([np.random.normal(loc=100., scale=np.sqrt(1.)), 0.], p=[0.1, 0.9])
            P_mat[s, a] = P_mat[s, a] / np.sum(P_mat[s, a])

    # R_mat
    for s in range(n_states):
        for a in range(n_actions):
            R_mat[s, a] += np.random.choice([np.random.normal(loc=1., scale=np.sqrt(0.1)), 0.], p=[0.5, 0.5])
            R_mat[s, a] += np.random.choice([np.random.normal(loc=100., scale=np.sqrt(1.)), 0.], p=[0.1, 0.9])
        R_mat[s] = 0.5 * R_mat[s] / np.max(R_mat[s])

    return P_mat, R_mat


def boltzmann_softmax(s_, Q, beta) -> float:
    n_actions = Q.shape[1]
    c = np.max(Q[s_])
    return np.sum([Q[s_][a_] * np.exp(beta * Q[s_][a_] - c) for a_ in range(n_actions)]) / np.sum(
        [np.exp(beta * Q[s_][a_] - c) for a_ in range(n_actions)])


def mellowmax(s_, Q, omega) -> float:
    n_states, n_actions = Q.shape[0], Q.shape[1]
    c = np.max(Q[s_])
    return c + np.log(np.sum([np.exp(omega * (Q[s_][a_] - c)) for a_ in range(n_actions)]) / float(len(Q[s_]))) / omega


# noinspection DuplicatedCode
def generalized_value_iteration_with_random_mdp(
        # n_mdp: int,
        all_mdp,
        n_trials: int,
        var_value: float,
        # Method: 'Boltzmann Softmax' or 'Mellowmax'
        method: str,
        # General settings
        save_dir: str,
        gamma: float = 0.98,
        log10_delta: int = -14.):
    """Plot vector field and distinct convergent points.

    :param all_mdp:
    :param n_mdp: Number of random MDPs to try
    :param n_trials: Number of trials for each random MDP
    :param method: Choose one from ['Boltzmann Softmax', 'Mellowmax']
    :param var_value: value of beta/omega
    :param save_dir:
    :param gamma: discount factor
    :param log10_delta: Criterion
    :return: No return. Only save plotted images.
    """
    n_mdp = len(all_mdp)

    # Method
    assert method in ['Boltzmann Softmax', 'Mellowmax']
    if method == 'Boltzmann Softmax':
        operator = boltzmann_softmax
    elif method == 'Mellowmax':
        operator = mellowmax
    else:
        raise ValueError(f'Invalid method: {method}.')

    # Collect data
    all_mdp_not_terminated = []
    all_mdp_not_single_fixed = []
    all_mdp_avg_iters = []

    # Random MDP loop
    mdp_pbar = tqdm(range(n_mdp), desc="All MDP", position=0)
    for mdp_idx in mdp_pbar:
        # Init
        # P, R = construct_mdp()
        P, R = all_mdp[mdp_idx]
        n_states, n_actions = P.shape[0], P.shape[1]
        # Record
        all_final_q = []
        n_not_terminated = 0
        total_iters = 0
        trial_pbar = tqdm(range(n_trials), desc="Trials of MDP", position=1, leave=False)
        for trial_idx in trial_pbar:
            # Beta loop: try diff beta; for each beta, plot an image
            # If using Mellowmax, please view beta here as omega
            # ================== Start ====================
            # Initialization
            # Q = np.random.rand(n_states, n_actions) * 0.7 + 0.3
            Q = np.random.rand(n_states, n_actions) * 30.
            # print('=======================================')
            # print('init', Q[0])
            # GVI loop
            n_max_iter = 2000
            done_iter = 0
            forced_terminated = False
            while True:
                # Init
                diff = np.full((n_states, n_actions), fill_value=-np.infty)
                # State loop (we only care about s1; s2 is a terminal state)
                for s in range(n_states):
                    # Action loop: iterate all actions
                    for a in range(n_actions):
                        # Q_copy <- Q(s,a)
                        Q_copy = deepcopy(Q[s][a])
                        # Q(s,a) <- R(s,a) + gamma * Summation_{s'} (P(s,a,s') * operator(...))
                        Q[s][a] = R[s][a] + gamma * sum([
                            P[s][a][s_] * (operator(s_, Q, var_value)) for s_ in range(n_states)])
                        # diff <- max(diff, cur_diff)
                        diff[s][a] = max(diff[s][a], np.log10(np.abs(Q_copy - Q[s][a]) + 1e-20))

                        # Record

                # Record
                done_iter += 1
                # If all action diff are < delta => stop
                # print()
                # print(f'doneiter={done_iter}, nmaxiter={n_max_iter}, cur_totaliter={total_iters}')
                # print(f'max_diff={np.max(diff)} {Q[0]}')
                if np.max(diff) < log10_delta:
                    break
                if done_iter >= n_max_iter:
                    forced_terminated = True
                    break
            if forced_terminated:
                n_not_terminated += 1
            total_iters += done_iter
            # =================== End =====================
            # Update progress bar
            # print('end', Q[0])
            all_final_q.append(Q)
            # print(f'Q => {Q}')
            trial_pbar.update(1)
        #
        all_mdp_not_terminated.append(n_not_terminated / float(n_trials))
        #
        all_final_q = np.array(all_final_q)
        all_final_q = np.round_(all_final_q, 5)
        uniq = np.unique(all_final_q, axis=0)
        all_mdp_not_single_fixed.append(1 if len(uniq) != 1 else 0)
        #
        # print()
        # print()
        # print()
        # print()
        # print(total_iters/float(n_trials), total_iters,n_trials)
        all_mdp_avg_iters.append(total_iters / float(n_trials))
        #
        mdp_pbar.update(1)
        # Print cur statistics
        mdp_pbar.set_postfix_str(
            f'notTer={np.mean(all_mdp_not_terminated)}; '
            f'notSingle={np.mean(all_mdp_not_single_fixed)}; '
            f'avgIter={np.mean(all_mdp_avg_iters)}')
        trial_pbar.close()
    mdp_pbar.close()


if __name__ == '__main__':
    SAVE_DIR = 'data/RandomMDP'
    ALL_MDP = [construct_mdp() for _ in range(200)]
    generalized_value_iteration_with_random_mdp(
        # n_mdp=200,
        all_mdp=ALL_MDP,
        n_trials=100,
        var_value=16.55,
        # Method: 'Boltzmann Softmax' or 'Mellowmax'
        method='Boltzmann Softmax',
        # method='Mellowmax',
        # General settings
        save_dir=SAVE_DIR,
        gamma=0.98,
        log10_delta=-14)
    generalized_value_iteration_with_random_mdp(
        # n_mdp=200,
        all_mdp=ALL_MDP,
        n_trials=100,
        var_value=16.55,
        # Method: 'Boltzmann Softmax' or 'Mellowmax'
        # method='Boltzmann Softmax',
        method='Mellowmax',
        # General settings
        save_dir=SAVE_DIR,
        gamma=0.98,
        log10_delta=-14)
