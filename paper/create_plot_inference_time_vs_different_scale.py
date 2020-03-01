import os
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig_name = 'msnet_inference_times.pdf'

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


fig, ax = plt.subplots()

# first row of plots
times = times.mean(1)
times = times*0.32/times.max()
ax.plot(times, 'bx')
ax.axvspan(-0.5, 3.5, alpha=0.5, color='gray')
ax.axvspan(3.5, 9.5, alpha=0.5, color='darkgrey')
ax.axvspan(9.5, 13.5, alpha=0.5, color='silver')
ax.axvspan(13.5, 14.5, alpha=0.5, color='lightgrey')
ax.set_ylabel('Execution time (sec)')

ax.axhspan(0.08, 0.33, alpha=0.3, color='darkgreen', label="max scale = 4")
ax.axhspan(0.012, 0.08, alpha=0.3, color='limegreen', label="max scale = 8")
ax.axhspan(-0.01, 0.012, alpha=0.3, color='lime', label="max scale $\geq$ 16")

# set x labels
ax.set_xticks(range(len(scales)))
ax.set_xticklabels(np.array(scales), rotation=80)
plt.legend()

fig.savefig(os.path.join(dir_latex_figures, fig_name), bbox_inches='tight')
plt.show(block=False)

