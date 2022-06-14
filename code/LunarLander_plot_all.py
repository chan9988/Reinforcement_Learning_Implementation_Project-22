import matplotlib.pyplot as plt
import torch
plt.rcParams.update({'font.size': 22})
curves = ['average', 'worst', 'best']
mello = True
if mello:
    name_ = 'mello'
    title_ = 'Mellowmax'
    title2_ = 'max entropy mellow'
    arr = [3, 5, 7, 8, 11]
    ccc = 'r'
    bo='\u03C9'
else:
    name_ = 'boltz'
    title_ = 'Boltzmann Softmax'
    title2_ = title_
    arr = [1, 2, 3, 5, 10]
    ccc = 'b'
    bo='\u03B2'
for curve_id in range(3):
    curve = curves[curve_id]
    _, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
    total_means = []
    for (beta, c) in zip(arr, ['b', 'g', 'r', 'c', 'm']):
        total_mean = 0
        ewmas_all = []
        means_all = []
        for seed in [22, 7654, 321]:
            ewmas = torch.load(f'ewma/ewma_{name_}_{beta}_b10_seed{seed}.pt')
            means = torch.load(f'mean/mean_{name_}_{beta}_b10_seed{seed}.pt')
            # print(len(means), beta,seed)
            # input()
            ewmas_all.append(ewmas)
            means_all.append(means)
            total_mean += means[-1]
        ewmas_all = torch.tensor(ewmas_all)
        means_all = torch.tensor(means_all)
        if curve_id == 0:
            total_means.append(total_mean/3)
            ewmas_all = ewmas_all.mean(dim=0)
            means_all = means_all.mean(dim=0)
        else:
            tmp = means_all[0]
            tmp2 = ewmas_all[0]
            for i, means in enumerate(means_all[1:, :]):
                if (curve_id == 1 and means.mean() < tmp.mean()) or (curve_id == 2 and means.mean() > tmp.mean()):
                    tmp = means
                    tmp2 = ewmas_all[i+1]
            means_all = tmp
            ewmas_all = tmp2
        # ax2.plot(ewmas_all, label=f'\u03C9={beta}', c=c)
        ax1.plot(means_all, label=f'{bo}={beta}', c=c,linewidth=8)
        ax1.yaxis.set_ticks(range(-500, 200,100))
    # ax2.set_title(f'{curve} {title_} ewma reward')
    # ax2.set_xlabel('episode')
    # ax2.set_ylabel(f'ewma reward')

    # ax1.set_title(f'{curve} {title_} mean retrun')
    ax1.set_xlabel('episode number')
    ax1.set_ylabel(f'mean return')
    ax1.legend()
    # ax2.legend()
    plt.savefig(f'result_curves/{name_}_{curve}.png')
    plt.close()
    if curve_id == 0:
        plt.figure(figsize=(15,10))
        plt.title(title2_)
        plt.ylim([20,120])
        plt.xticks(arr)
        plt.yticks(range(20, 160, 20))
        plt.xlabel(bo)  
        plt.ylabel('mean return')
        plt.plot(arr, total_means, c=ccc,linewidth=8)
        plt.savefig(f'result_curves/{name_}_cmp.png')
        plt.close()
