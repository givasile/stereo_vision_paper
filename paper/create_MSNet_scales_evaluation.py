import os
import configparser
import matplotlib.pyplot as plt
import numpy as np
import torch
import helping_functions as hf
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig_name = 'msnet_scales_evaluation.pdf'

conf_path = './../conf.ini'
conf = configparser.ConfigParser()
conf.read(os.path.abspath(conf_path))

dir_saved_models = conf['PATHS']['saved_models']
dir_latex_figures = os.path.abspath('./latex/figures/')

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

# plot
fig, ax = plt.subplots(1, 2, sharex='all')
# fig.suptitle("MSNet evaluation using different combinations of scales")

# first row of plots
ax[0].plot(results[:,0], 'ro')
ax[0].axvspan(-0.5, 3.5, alpha=0.5, color='gray')
ax[0].axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
ax[0].axvspan(9.5, 13.5, alpha=0.5, color='silver')
ax[0].axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
ax[0].set_ylabel('$\mu$ Absolute Error')

ax[1].plot(results[:,1], 'ro')
ax[1].axvspan(-0.5, 3.5, alpha=0.5, color='gray')
ax[1].axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
ax[1].axvspan(9.5, 13.5, alpha=0.5, color='silver')
ax[1].axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
ax[1].set_ylabel('$\sigma$ Absolute Error')

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
xnames = ['\{4\}', '\{8\}', '\{16\}', '\{32\}',
          '\{4,8\}', '\{4,16\}', '\{4,32\}', '\{8,16\}', '\{8, 32\}', '\{16,32\}',
          '\{4,8,16\}','\{4,8,32\}','\{4,16,32\}','\{8,16,32\}',
          '\{4,8,16,32\}']
ax[0].set_xticks(range(len(results[:,3])))
ax[0].set_xticklabels(xnames, rotation=80)
ax[1].set_xticklabels(xnames, rotation=80)
# ax[1][0].set_xticklabels(xnames, rotation=80)
# ax[1][1].set_xticklabels(xnames, rotation=80)

fig.savefig(os.path.join(dir_latex_figures, fig_name), bbox_inches='tight')
plt.show(block=False)
