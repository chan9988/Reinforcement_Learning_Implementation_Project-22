from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import glob
import os
from PIL import Image

# === Environment settings ===
# We only care about s1 since s2 is a terminal state.
# R = {'s1': {'a': {'s1': 0.122,
#                   's2': 0.122},
#             'b': {'s1': 0.033,
#                   's2': 0.033}}}
R = {'s1': {'a': 0.122,
            'b': 0.033}}
P = {'s1': {'a': {'s1': 0.66,
                  's2': 0.34},
            'b': {'s1': 0.99,
                  's2': 0.01}}}
ALL_ACTIONS = ['a', 'b']


def make_gif(source_img_dir):
    """
    Reference:
    https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    :param source_img_dir: 
    :return: 
    """
    # File path
    fp_in = os.path.join(source_img_dir, "*.png")
    fp_out = os.path.join(source_img_dir, "GIF.gif")

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)


def boltzmann_softmax(_s_, _Q_a, _Q_b, beta) -> float:
    assert _s_ in ['s1', 's2']
    if _s_ == 's2':
        return 0.
    elif _s_ == 's1':
        c = max(_Q_a, _Q_b)
        pa = math.exp(beta * _Q_a - c) / (math.exp(beta * _Q_a - c) + math.exp(beta * _Q_b - c))
        pb = math.exp(beta * _Q_b - c) / (math.exp(beta * _Q_a - c) + math.exp(beta * _Q_b - c))
        # print(f'p1 p2 = {p1:.4f} {p2:.4f}', _Q_a, _Q_b)
        ret = pa * _Q_a + pb * _Q_b
        # print(f'boltz: {ret}')
        return ret


def mellowmax(_s_, _Q_a, _Q_b, omega) -> float:
    assert _s_ in ['s1', 's2']
    if _s_ == 's2':
        return 0.
    elif _s_ == 's1':
        c = max(_Q_a, _Q_b)
        mm = (math.exp(omega * (_Q_a - c)) + math.exp(omega * (_Q_b - c))) / 2
        mm = c + math.log(mm) / omega
        return mm


def generalized_value_iteration(
        # Method: 'Boltzmann Softmax' or 'Mellowmax'
        method: str,
        # General settings
        save_dir: str,
        gamma: float = 0.98,
        log10_delta: int = -15,
        # Choose what to plot
        plot_first_vec: bool = True,
        plot_final_vec: bool = True,
        # Range of beta/omega, Q_a, Q_b to iterate
        start_beta: float = 16.874,
        end_beta: float = 16.880,
        step_beta: float = 0.001,
        start_a: float = 0.30,
        end_a: float = 0.80,
        step_a: float = 0.1,
        start_b: float = 0.425,
        end_b: float = 0.925,
        step_b: float = 0.1,
        # Plot settings
        figsize: tuple = (10, 10),
        vec_len_reduce: float = 2,
        first_arr_width: float = 0.0010,
        final_arr_width: float = 0.002):
    """Plot vector field and distinct convergent points.

    :param method: Choose one from ['Boltzmann Softmax', 'Mellowmax']
    :param save_dir:
    :param gamma: discount factor
    :param log10_delta: Criterion
    :param plot_first_vec: Whether plot first update vector
    :param plot_final_vec: Whether plot vector from init point to final point
    :param start_beta: beta loop start value
    :param end_beta: (inclusive)
    :param step_beta: step
    :param start_a: Q_a loop start value
    :param end_a: (inclusive)
    :param step_a: step
    :param start_b: Q_b loop start value
    :param end_b: (inclusive)
    :param step_b: step
    :param figsize: plt figsize
    :param vec_len_reduce: divide vec len by this value
    :param first_arr_width: first update vec width
    :param final_arr_width: final update vec width
    :return: No return. Only save plotted images.
    """
    # Plot settings
    assert plot_first_vec or plot_final_vec, 'Must choose at least one.'
    var_name = 'beta' if method == 'Boltzmann Softmax' else 'omega'

    # Method
    assert method in ['Boltzmann Softmax', 'Mellowmax']
    if method == 'Boltzmann Softmax':
        operator = boltzmann_softmax
    elif method == 'Mellowmax':
        operator = mellowmax
    else:
        raise ValueError(f'Invalid method: {method}.')

    # Loop range
    beta_arange = np.arange(start_beta, end_beta, step_beta)
    init_qa_arange = np.arange(start_a, end_a + step_a, step_a)
    init_qb_arange = np.arange(start_b, end_b + step_b, step_b)
    total_gvi = len(beta_arange) * len(init_qa_arange) * len(init_qb_arange)
    loop_bar = tqdm(None, total=total_gvi, desc='Total GVI')

    # Collect data
    all_beta = []
    all_distinct_final_pair = []

    # Beta loop: try diff beta; for each beta, plot an image
    # If using Mellowmax, please view beta here as omega
    for beta_idx, beta in enumerate(beta_arange):
        # Collect data
        # - init: init points
        # - first: first update points
        # - final: final update points
        all_init_qa = []
        all_init_qb = []
        all_first_update_qa = []
        all_first_update_qb = []
        all_final_qa = []
        all_final_qb = []
        all_final_q_pair = []
        # Init Q_a loop
        for init_qa in init_qa_arange:
            # Init Q_b loop
            for init_qb in init_qb_arange:
                # Initialization
                Q = {'a': init_qa,
                     'b': init_qb}
                all_init_qa.append(init_qa)
                all_init_qb.append(init_qb)
                y_q = {'a': [], 'b': []}
                first_update_qa = True
                first_update_qb = True
                # GVI loop
                while True:
                    # Init
                    ok = False  # If only plot first update, it breaks after first update
                    diff = {'a': -99999999999999999, 'b': -99999999999999999}
                    # State loop (we only care about s1; s2 is a terminal state)
                    for s in ['s1']:
                        # Action loop: iterate all actions
                        for a in ALL_ACTIONS:
                            # Q_copy <- Q(s,a)
                            Q_copy = deepcopy(Q[a])
                            # Q(s,a) <- R(s,a) + gamma * Summation_{s'} (P(s,a,s') * operator(...))
                            Q[a] = R[s][a] + gamma * sum(
                                [P[s][a][s_] * (operator(s_, Q['a'], Q['b'], beta)) for s_ in
                                 ['s1', 's2']])
                            # diff <- max(diff, cur_diff)
                            diff[a] = max(diff[a], np.log10(np.abs(Q_copy - Q[a])))

                            # Record
                            y_q[a].append(Q[a])

                            # Update first_update variables
                            if a == 'a' and first_update_qa:
                                first_update_qa = False
                                all_first_update_qa.append(Q[a])
                            if a == 'b' and first_update_qb:
                                first_update_qb = False
                                all_first_update_qb.append(Q[a])

                            # Break if only plot first updates
                            if not first_update_qb and (not first_update_qa):
                                if not plot_final_vec:
                                    ok = True
                                    break
                    # Record
                    for a in ALL_ACTIONS:
                        y_q[a].append(Q[a])

                    # If all action diff are < delta => stop
                    if max([diff[a] for a in ALL_ACTIONS]) < log10_delta:
                        break

                    # Stop if there's custom stop command
                    if ok:
                        break
                # Record
                all_final_qa.append(Q["a"])
                all_final_qb.append(Q["b"])
                all_final_q_pair.append((Q['a'], Q['b']))
                # Update progress bar
                loop_bar.update(1)
        # Record distinct convergent points
        all_beta.append(beta)
        all_distinct_final_pair.append(list(set(all_final_q_pair)))

        # Plot this beta's result as an image
        plt.figure(figsize=figsize)
        for qa, qb, up_qa, up_qb, final_qa, final_qb in zip(all_init_qa, all_init_qb, all_first_update_qa,
                                                            all_first_update_qb, all_final_qa, all_final_qb):
            first_da = (up_qa - qa) / vec_len_reduce  # first step
            first_db = (up_qb - qb) / vec_len_reduce  # first step
            final_da = (final_qa - qa) / vec_len_reduce  # diff between init and final
            final_db = (final_qb - qb) / vec_len_reduce  # diff between init and final
            if plot_final_vec:
                plt.arrow(qb, qa, final_db, final_da, width=final_arr_width, color='green')
            if plot_first_vec:
                plt.arrow(qb, qa, first_db, first_da, width=first_arr_width, color='blue')
        # Image title
        title_str = f'GVI ({method})'
        if plot_final_vec and not plot_first_vec:
            title_str += f' ({var_name}={beta:.5f}) (green=final)'
        elif not plot_final_vec and plot_first_vec:
            title_str += f' ({var_name}={beta:.5f}) ("blue=first")'
        elif plot_final_vec and plot_first_vec:
            title_str += f' ({var_name}={beta:.5f}) (green=final;blue=first)'
        else:
            raise ValueError('Must plot at least one type.')
        title_str += f' (vecLenReduce={vec_len_reduce}) '
        plt.title(title_str)
        # Image axis labels
        plt.ylabel(f'Q(s1,a)')
        plt.xlabel(f'Q(s1,b)')
        plt.savefig(os.path.join(save_dir, f'{method}_{beta_idx:>010}.png'))
        plt.clf()
    # Close the progress bar
    loop_bar.close()
    # Plot beta/omega - Q(s1,a/b) images
    assert len(all_beta) == len(all_distinct_final_pair)
    qa_x = []
    qa_y = []
    qb_x = []
    qb_y = []
    for beta_i, beta in enumerate(all_beta):
        for pair in all_distinct_final_pair[beta_i]:
            qa_x.append(beta)
            qa_y.append(pair[0])
            qb_x.append(beta)
            qb_y.append(pair[1])
    plt.title(f'Fixed Points under {method} with {var_name}')
    plt.scatter(qa_x, qa_y, label='Q(s1,a)')
    plt.scatter(qb_x, qb_y, label='Q(s1,b)')
    plt.xlabel(f'{var_name}')
    plt.ylabel(f'Q value')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{method}_fixed_points.png'))
    plt.clf()


if __name__ == '__main__':
    SAVE_DIR = 'data/sample_mdp/GVI'
    generalized_value_iteration(
        # Method: 'Boltzmann Softmax' or 'Mellowmax'
        method='Boltzmann Softmax',
        # method='Mellowmax',
        # General settings
        save_dir=SAVE_DIR,
        gamma=0.98,
        log10_delta=-15,
        # Choose what to plot
        plot_first_vec=True,
        plot_final_vec=True,
        # Range of beta to iterate (if you use mellowmax, view beta as omega here)
        start_beta=16.0,
        end_beta=18.0,
        step_beta=0.02,
        # Range of init Q_a to iterate
        start_a=0.30,
        end_a=0.90,
        step_a=0.1,
        # Range of init Q_b to iterate
        start_b=0.40,
        end_b=1.0,
        step_b=0.1,
        # Plot settings
        figsize=(10, 10),
        vec_len_reduce=2,
        first_arr_width=0.0010,
        final_arr_width=0.002)
    # Make GIF
    make_gif(SAVE_DIR)
