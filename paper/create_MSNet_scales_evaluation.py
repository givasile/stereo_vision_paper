import os
import configparser
import matplotlib.pyplot as plt
import numpy as np
import torch
import helping_functions as hf
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig_name1 = 'msnet_scales_evaluation1.pdf'
fig_name2 = 'msnet_scales_evaluation2.pdf'

conf_path = './../conf.ini'
conf = configparser.ConfigParser()
conf.read(os.path.abspath(conf_path))

dir_saved_models = conf['PATHS']['saved_models']
dir_latex_figures = os.path.abspath('./latex/figures/')

# compute path to all experiments checkpoints
with open('./saved_info/times_for_different_scale.p', 'rb') as fm:
    results_dict = pickle.load(fm)

scales = results_dict['scales']
times = results_dict['times']

# compute path to all experiments checkpoints
dir_scalable_net = os.path.join(dir_saved_models, 'vol2/merging_info_net_custom_features')
experiments = []
for i in range(101, 116):
    experiments.append({'id': i,
                        'path': os.path.join(dir_scalable_net, 'experiment_' + str(i), 'checkpoint_1.tar')
                        })

# compute the statistics of all experiments
results = []
for i, exp in enumerate(experiments):
    stats = torch.load(exp['path'])['stats']
    results.append([
        np.mean(stats['test_full']['mae']['0_after'][0]),
        np.std(stats['test_full']['mae']['0_after'][0]),
        np.mean(stats['test_full']['pcg']['0_after'][0]),
        np.std(stats['test_full']['pcg']['0_after'][0])
    ])
results = np.array(results)

# fig 1
fig, ax = plt.subplots()
ax.plot(results[:,0], 'ro')
ax.axvspan(-0.5, 3.5, alpha=0.5, color='gray')
ax.axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
ax.axvspan(9.5, 13.5, alpha=0.5, color='silver')
ax.axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
ax.set_ylabel('$\mu$ Absolute Error')


xnames = ['\{4\}', '\{8\}', '\{16\}', '\{32\}',
          '\{4,8\}', '\{4,16\}', '\{4,32\}', '\{8,16\}', '\{8, 32\}', '\{16,32\}',
          '\{4,8,16\}','\{4,8,32\}','\{4,16,32\}','\{8,16,32\}',
          '\{4,8,16,32\}']
ax.set_xticks(range(len(results[:,3])))
ax.set_xticklabels(xnames, rotation=80)
fig.savefig(os.path.join(dir_latex_figures, fig_name1), bbox_inches='tight')

# fig 2
fig, ax = plt.subplots()
ax.plot(results[:,1], 'ro')
ax.axvspan(-0.5, 3.5, alpha=0.5, color='gray')
ax.axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
ax.axvspan(9.5, 13.5, alpha=0.5, color='silver')
ax.axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
ax.set_ylabel('$\sigma$ Absolute Error')
ax.set_xticks(range(len(results[:,3])))
ax.set_xticklabels(xnames, rotation=80)
fig.savefig(os.path.join(dir_latex_figures, fig_name2), bbox_inches='tight')

# times = times.mean(1)
# ax[2].plot(times, 'rx')
# ax[2].axvspan(-0.5, 3.5, alpha=0.5, color='gray')
# ax[2].axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
# ax[2].axvspan(9.5, 13.5, alpha=0.5, color='silver')
# ax[2].axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
# ax[2].set_ylabel('$\sigma$ Absolute Error')


# second row of plots
# ax[1][0].plot(results[:,2], 'ro')
# ax[1][0].axvspan(-0.5, 3.5, alpha=0.5, color='blue')
# ax[1][0].axvspan(3.5, 9.5, alpha=0.5, color='green')
# ax[1][0].axvspan(9.5, 13.5, alpha=0.5, color='magenta')
# ax[1][0].axvspan(13.5, 14.5, alpha=0.5, color='red')
# ax[1][0].set_ylabel('$\mu$ PCG Error')

# ax[1][1].plot(results[:,3], 'ro')
# ax[1][1].axvspan(-0.5, 3.5, alpha=0.5, color='blue')
# ax[1][1].axvspan(3.5, 9.5, alpha=0.5, color='green')
# ax[1][1].axvspan(9.5, 13.5, alpha=0.5, color='magenta')
# ax[1][1].axvspan(13.5, 14.5, alpha=0.5, color='red')
# ax[1][1].set_ylabel('$\sigma$ PCG Error')

# set x labels
# xnames = ['\{4\}', '\{8\}', '\{16\}', '\{32\}',
#           '\{4,8\}', '\{4,16\}', '\{4,32\}', '\{8,16\}', '\{8, 32\}', '\{16,32\}',
#           '\{4,8,16\}','\{4,8,32\}','\{4,16,32\}','\{8,16,32\}',
#           '\{4,8,16,32\}']
# ax[0].set_xticks(range(len(results[:,3])))
# ax[0].set_xticklabels(xnames, rotation=80)
# ax[1].set_xticklabels(xnames, rotation=80)
# # ax[1][0].set_xticklabels(xnames, rotation=80)
# # ax[1][1].set_xticklabels(xnames, rotation=80)

# fig.savefig(os.path.join(dir_latex_figures, fig_name), bbox_inches='tight')
# plt.show(block=False)
