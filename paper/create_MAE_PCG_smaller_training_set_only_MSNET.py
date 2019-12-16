import os
import configparser
import matplotlib.pyplot as plt
import numpy as np
import torch
import helping_functions as hf


conf_path = './../conf.ini'
conf = configparser.ConfigParser()
conf.read(os.path.abspath(conf_path))

dir_saved_models = conf['PATHS']['saved_models']
dir_latex_figures = os.path.abspath('./latex/figures/')

cnn_name_mapping = dict(conf['CNN_NAME_MAPPING'])

# freiburg paths
tmp = hf.prepend_path(cnn_name_mapping, 'vol2')
tmp1 = {}
for key, val in tmp.items():
    if key.startswith('scalable'):
        tmp1[key] = val
tmp1 = hf.prepend_path(tmp1, dir_saved_models)

freiburg_last_checkpoint_1 = hf.find_last_checkpoint(hf.append_path(tmp1, 'experiment_1'))
freiburg_standard_checkpoint_1 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_1'), 10)
freiburg_last_checkpoint_21 = hf.find_last_checkpoint(hf.append_path(tmp1, 'experiment_21'))
freiburg_last_checkpoint_23 = hf.find_last_checkpoint(hf.append_path(tmp1, 'experiment_23'))
freiburg_last_checkpoint_25 = hf.find_last_checkpoint(hf.append_path(tmp1, 'experiment_25'))
freiburg_last_checkpoint_27 = hf.find_last_checkpoint(hf.append_path(tmp1, 'experiment_27'))



def plot(fig_name, title, mae_or_pcg, tr_te_val):
    cnn_name = 'scalable_net'

    # figure 1
    fig, ax = plt.subplots(1, 1)

    # load stats
    def add_plot(last_checkpoint, label, formating = None):
        stats = torch.load(last_checkpoint[cnn_name])['stats']
        lis = stats[tr_te_val][mae_or_pcg]['0_after']
        y = [np.mean(lis[i]) for i in range(len(lis))]
        x = np.arange(1, len(y) + 1)
        if formating is not None:
            ax.plot(x, y, formating, label=label)
        else:
            ax.plot(x, y, '-o', label=label)

    add_plot(freiburg_last_checkpoint_21, "10%")
    add_plot(freiburg_last_checkpoint_23, "30%")
    add_plot(freiburg_last_checkpoint_25, "50%")
    add_plot(freiburg_last_checkpoint_27, "70%")
    add_plot(freiburg_standard_checkpoint_1, "100%", 'r-o')

    fig.suptitle(title)
    ax.set_ylabel("pixels") if mae_or_pcg == "mae" else ax.set_ylabel("pcg")
    ax.set_xlabel("training epoch")
    ax.legend()
    fig.savefig(os.path.join(dir_latex_figures, fig_name), bbox_inches='tight')
    plt.show()


# MAE
fig_name = "freiburg_msnet_mae_smaller_training_set.pdf"
title = "MSNET MAE on smaller training set"
plot(fig_name, title, "mae", "test_full")

# PCG
fig_name = "freiburg_msnet_pcg_smaller_training_set.pdf"
title = "MSNET PCG on smaller training set"
plot(fig_name, title, "pcg", "test_full")
