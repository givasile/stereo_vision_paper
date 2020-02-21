'''
Script for creating plot with:
* x: the pcg of Freiburg training set used
* y: the MAE (PCG) after 10 epochs of training
* for MSNET and Free Weight cnns
'''


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

# get latest checkpoint
freiburg_checkpoint_1 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_1'), 3)
freiburg_checkpoint_21 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_21'), 3)
freiburg_checkpoint_23 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_23'), 3)
freiburg_checkpoint_25 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_25'), 3)
freiburg_checkpoint_27 = hf.choose_specific_checkpoint(hf.append_path(tmp1, 'experiment_27'), 3)


def plot(fig_name, title, mae_or_pcg, tr_te_val):
    cnn_to_plot = ['scalable_net',
                   'scalable_net_free_2d',
                   'scalable_net_free_3d',
                   'scalable_net_free_2d_3d']

    # figure 1
    fig, ax = plt.subplots(1, 1)

    for cnn_name in cnn_to_plot:
        x = []
        y = []

        def add_results_to_x_y(pcg, checkpoint):
            if cnn_name in checkpoint.keys():
                if int(checkpoint[cnn_name][-5]) != 1:  # ugly hack for assuring it is not checkpoint_1.tar
                    stats = torch.load(checkpoint[cnn_name])['stats']
                    last_epoch = stats[tr_te_val][mae_or_pcg]['0_after'][-1]
                    x.append(pcg*100)
                    y.append(np.mean(last_epoch))

        add_results_to_x_y(0.1, freiburg_checkpoint_21)
        add_results_to_x_y(0.3, freiburg_checkpoint_23)
        add_results_to_x_y(0.5, freiburg_checkpoint_25)
        add_results_to_x_y(0.7, freiburg_checkpoint_27)
        add_results_to_x_y(1, freiburg_checkpoint_1)

        ax.plot(x, y, hf.coloring[cnn_name] + '-o', label=hf.labeling[cnn_name])

    # fig.suptitle(title)
    ax.set_ylabel("pixels") if mae_or_pcg == "mae" else ax.set_ylabel("pcg")
    ax.set_xlabel("pcg of training set ")
    ax.legend()
    plt.xticks([10, 30, 50, 70, 100])
    fig.savefig(os.path.join(dir_latex_figures, fig_name), bbox_inches='tight')
    plt.show(block=False)

# MAE
fig_name = "freiburg_msnet_vs_free_weights_mae_smaller_tr_set.pdf"
title = "MSNET vs Free Weights: MAE in smaller training set"
plot(fig_name, title, "mae", "test_full")

# PCG
fig_name = "freiburg_msnet_vs_free_weights_pcg_smaller_tr_set.pdf"
title = "MSNET vs Free Weights: PCG in smaller training set"
plot(fig_name, title, "pcg", "test_full")


