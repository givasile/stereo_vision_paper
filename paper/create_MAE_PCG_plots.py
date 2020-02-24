import os
import configparser
import helping_functions as hf

conf_path = './../conf.ini'
conf = configparser.ConfigParser()
conf.read(os.path.abspath(conf_path))

dir_saved_models = conf['PATHS']['saved_models']
dir_latex_figures = os.path.abspath('./latex/figures/')

cnn_name_mapping = dict(conf['CNN_NAME_MAPPING'])

# freiburg paths
tmp = hf.prepend_path(cnn_name_mapping, 'vol2')
tmp = hf.prepend_path(tmp, dir_saved_models)
tmp = hf.append_path(tmp, 'experiment_1')

freiburg_last_checkpoint = hf.find_last_checkpoint(tmp)
freiburg_standard_checkpoint = hf.choose_specific_checkpoint(tmp, 10)

# kitti 2015 paths
tmp = hf.prepend_path(cnn_name_mapping, 'vol2')
tmp = hf.prepend_path(tmp, dir_saved_models)
tmp = hf.append_path(tmp, 'experiment_2')

kitti_2015_last_checkpoint = hf.find_last_checkpoint(tmp)
kitti_2015_standard_checkpoint = hf.choose_specific_checkpoint(tmp, 200)

# kitti 2012 paths
tmp = hf.prepend_path(cnn_name_mapping, 'vol2')
tmp = hf.prepend_path(tmp, dir_saved_models)
tmp = hf.append_path(tmp, 'experiment_3')

kitti_2012_last_checkpoint = hf.find_last_checkpoint(tmp)
kitti_2012_standard_checkpoint = hf.choose_specific_checkpoint(tmp, 75)

# Freiburg

fig_name = 'freiburg_msnet_vs_monolithic_mae.pdf'
title = "Synthetic Flying Dataset MAE - MSNet vs Monolithic"
checkpoints = freiburg_standard_checkpoint
mae_or_pcg = "mae"
tr_te_val = "test_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'freiburg_msnet_vs_monolithic_pcg.pdf'
title = "Synthetic Flying Dataset PCG - MSNet vs Monolithic"
checkpoints = freiburg_standard_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "test_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'freiburg_msnet_vs_free_weights_mae.pdf'
title = "Synthetic Flying Dataset MAE - MSNet vs Free Weights"
checkpoints = freiburg_last_checkpoint
mae_or_pcg = "mae"
tr_te_val = "test_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'freiburg_msnet_vs_free_weights_pcg.pdf'
title = "Synthetic Flying Dataset PCG - MSNet vs Free Weights"
checkpoints = freiburg_last_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "test_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)

# KITTI 2015

fig_name = 'kitti2015_msnet_vs_monolithic_mae.pdf'
title = "KITTI 2015 MAE - MSNet vs Monolithic"
checkpoints = kitti_2015_last_checkpoint
mae_or_pcg = "mae"
tr_te_val = "val_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2015_msnet_vs_monolithic_pcg.pdf'
title = "KITTI 2015 PCG - MSNet vs Monolithic"
checkpoints = kitti_2015_last_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "val_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2015_msnet_vs_free_weights_mae.pdf'
title = "KITTI 2015 MAE - MSNet vs Free Weights"
checkpoints = kitti_2015_last_checkpoint
mae_or_pcg = "mae"
tr_te_val = "val_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2015_msnet_vs_free_weights_pcg.pdf'
title = "KITTI 2015 PCG - MSNet vs Free Weights"
checkpoints = kitti_2015_last_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "val_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)

# KITTI 2012

fig_name = 'kitti2012_msnet_vs_monolithic_mae.pdf'
title = "KITTI 2012 MAE - MSNet vs Monolithic"
checkpoints = kitti_2012_last_checkpoint
mae_or_pcg = "mae"
tr_te_val = "val_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2012_msnet_vs_monolithic_pcg.pdf'
title = "KITTI 2012 PCG - MSNet vs Monolithic"
checkpoints = kitti_2012_last_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "val_full"
hf.plot_msnet_vs_monolithic(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2012_msnet_vs_free_weights_mae.pdf'
title = "KITTI 2012 MAE - MSNet vs Free Weights"
checkpoints = kitti_2012_last_checkpoint
mae_or_pcg = "mae"
tr_te_val = "val_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)


fig_name = 'kitti2012_msnet_vs_free_weights_pcg.pdf'
title = "KITTI 2012 PCG - MSNet vs Free Weights"
checkpoints = kitti_2012_last_checkpoint
mae_or_pcg = "pcg"
tr_te_val = "val_full"
hf.plot_msnet_vs_free_weights(fig_name, title, checkpoints, mae_or_pcg, tr_te_val)

