'''
Script for figure 3 of the paper, where it is proven with a real case scenario 
(2 specific points are choosen) the importance of multiscale analysis.

'''
import sys
import configparser
import importlib
import torch
import timeit
import os
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use("TkAgg")


# import config file
conf_path = '../conf.ini'
conf_file = configparser.ConfigParser()
conf_file.read(conf_path)

# add parent path, if not already added
parent_path = conf_file['PATHS']['PARENT_DIR']
sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

################# Configuration ####################
cnn_name = 'merging_info_net_custom_features'
save_fig = True
high_low = 'high'

# directory to load_from
experiment_n = 1      # directory to load_from
checkpoint_n = 15                # which checkpoint to load weights/stats from
which = 'test'
form = 'full_im'
limit_maxD = True
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 100
device = 'cuda'

net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
preprocess = importlib.import_module('preprocess')
utils = importlib.import_module('vol2.models.utils')

# create helpful paths
experiment_directory = os.path.join(conf_file['PATHS']['saved_models'], 'vol2', cnn_name,
                                    'experiment_' + str(experiment_n))
checkpoint_filepath = os.path.join(experiment_directory, 'checkpoint_' + str(checkpoint_n) + '.tar')

# load dataset
if get_common_dataset:
    merged_dataset = utils.load_common_dataset(common_dataset_name, conf_file['PATHS']['common_datasets'])
else:
    merged_dataset = utils.load_specific_dataset(experiment_directory)

# create model
model_instance = net.model().cuda()

# restore weights
checkpoint = torch.load(checkpoint_filepath)
model_instance.load_state_dict(checkpoint['state_dict'])

# restore training statistics
stats = checkpoint['stats']

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model_instance.parameters()])))

# input
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.type(torch.bool).unsqueeze(0).cuda()

# inspection
max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]

def predict(scales, prediction_from_scales):
    tmp = timeit.default_timer()
    ret = net.inspection(
        model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)
    print("Inspection execution time: %s" % (timeit.default_timer()-tmp))
    return ret

# multiscale prediction
scales = [[round(max_disp/4), round(h/4), round(w/4)],
          [round(max_disp/8), round(h/8), round(w/8)],
          [round(max_disp/16), round(h/16), round(w/16)],
          [round(max_disp/32), round(h/32), round(w/32)]]
prediction_from_scales = {3: ['after'],
                          2: ['after'],
                          1: ['after'],
                          0: ['after']}
mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = predict(
    scales, prediction_from_scales)

# single scale prediction
single_scale = []
for sc in [4, 8, 16, 32]:
    scales = [[round(max_disp/sc), round(h/sc), round(w/sc)]]
    prediction_from_scales = {0: ['after']}
    single_scale.append(predict(scales, prediction_from_scales))
    

### VISUALIZATION ###
step = 1

# example where low resolution matters
x_low = 300
y_low = 440
# x_low = 452
# y_low = 664
# example where only high resolution matters
x_high = 237
y_high = 573
# x_high = 200
# y_high = 430


if high_low == 'low':
    x = x_low
    y = y_low
elif high_low == 'high':
    x = x_high
    y = y_high

gt = dispL[:, x, y].item()
gt_low = dispL[:, x_low, y_low].item()
gt_high = dispL[:, x_high, y_high].item()

# multiscale 
scales = [4, 8, 16, 32]
x_init = []
y_init = []
x_new = []
y_new = []
predict = []
for i, sc in enumerate(scales):
    similarity_volume = for_out_dict[i]['after'][0,0, :, round(x/sc), round(y/sc)].cpu().numpy()
    similarity_volume = softmax(similarity_volume)
    x_init.append(np.arange(0, max_disp+1, sc))
    y_init.append(similarity_volume)
    x_new.append(np.arange(0,max_disp+1, step))
    y_new.append(np.interp(x_new[i], x_init[i], y_init[i]))
    predict.append(predictions_dict[i]['after'][0,x,y].item())
    
# single scale
scales = [4, 8, 16, 32]
x_init_single = []
y_init_single = []
x_new_single = []
y_new_single = []
predict_single = []
for i, sc in enumerate(scales):
    similarity_volume_single = single_scale[i][6][0]['after'][0,0, :, round(x/sc), round(y/sc)].cpu().numpy()
    similarity_volume_single = softmax(similarity_volume_single)
    x_init_single.append(np.arange(0, max_disp+1, sc))
    y_init_single.append(similarity_volume_single)
    x_new_single.append(np.arange(0,max_disp+1, step))
    y_new_single.append(np.interp(x_new_single[i], x_init_single[i], y_init_single[i]))
    predict_single.append(single_scale[i][7][0]['after'][0,x,y].item())
    

fig = plt.figure(figsize = (7,2))

font = {'family': 'serif',
        'color':  'darkslategrey',
        'weight': 'normal',
        'size': 9,
        }

# first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
i = 0
for c, z in zip(['c', 'c', 'c', 'c'], [0, 1, 2, 3]):
    # compute x,y,z
    xs = x_new[0]
    ys = y_new[i]
    if ys[round(predict[i]/step)] < 0.3:
        ys[round(predict[i]/step)] = 0.3
        
    if ys[round(gt/step)] < 0.3:
        ys[round(gt/step)] = 0.3

    # set color
    cs = [c] * len(xs)
    cs[round(gt/step)] = 'r'
    cs[round(predict[i]/step)] = 'b'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.75)
    
    i += 1

ax.set_xlabel('disparity', fontdict = font, labelpad = -7.5, x=-10)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels([str(0), str(20), str(40), str(60), str(80)], fontdict = font)
ax.set_xlim3d(0, 100)
ax.xaxis.set_tick_params(pad = -5, labelsize = 8)

ax.set_ylabel('scale', fontdict = font, labelpad = 2)
ax.set_yticks([1,2,3,4])
ax.set_yticklabels(['$\{2^2,...,2^5\}$',
                    '$\{2^3,2^4,2^5\}$',
                    '$\{2^4,2^5\}$',
                    '$\{2^5\}$'], fontdict=font)
ax.yaxis.set_tick_params(pad = -1, labelsize = 5, rotation=-10)

ax.set_zlabel('probability', fontdict = font, labelpad = -25)
ax.set_zticks([0.5, 1])
# ax.set_zticklabels([str(0.5), str(1)], font)
ax.zaxis.set_tick_params(pad = -2, labelsize = 8, labelcolor = 'darkslategrey')

if high_low == 'high':
    ax.legend(handles = [Patch(facecolor='none', edgecolor='yellow', label='multi-scale')], fontsize=7)
elif high_low == 'low':
    ax.legend(handles = [Patch(facecolor='none', edgecolor='orange', label='multi-scale')], fontsize=7)

# second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
i = 0
for c, z in zip(['c', 'c', 'c', 'c'], [1, 2, 3, 4]):
    xs = x_new_single[0]
    ys = y_new_single[i]
    if ys[round(predict_single[i]/step)] < 0.3:
        ys[round(predict_single[i]/step)] = 0.3
    if ys[round(gt/step)] < 0.3:
        ys[round(gt/step)] = 0.3

    # set color
    cs = [c] * len(xs)
    cs[round(gt/step)] = 'r'
    cs[round(predict_single[i]/step)] = 'b'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.75)
    
    i += 1

ax.set_xlabel('disparity', fontdict = font, labelpad = -7.5, x=-10)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels([str(0), str(20), str(40), str(60), str(80)], fontdict = font)
ax.set_xlim3d(0, 100)
ax.xaxis.set_tick_params(pad = -5, labelsize = 8)

ax.set_ylabel('scale', fontdict = font, labelpad = -8)
ax.set_yticks([1,2,3,4])
ax.set_yticklabels(['$2^2$','$2^3$','$2^4$','$2^5$'], fontdict=font)
ax.yaxis.set_tick_params(pad = -3, labelsize = 7)

ax.set_zlabel('probability', fontdict = font, labelpad = -25)
ax.set_zticks([0.5, 1])
# ax.set_zticklabels([str(0.5), str(1)], font)
ax.zaxis.set_tick_params(pad = -2, labelsize = 8, labelcolor = 'darkslategrey')

if high_low == 'high':
    ax.legend(handles = [Patch(facecolor='none', edgecolor='yellow', label='single scale')], fontsize=7)
elif high_low == 'low':
    ax.legend(handles = [Patch(facecolor='none', edgecolor='orange', label='single scale')], fontsize=7)

# plt.tight_layout()
if save_fig:
    if high_low == 'low':
        plt.savefig('./latex/figures/multiscale_importance_graph_low_resolution.pdf', bbox_inches = 'tight')
    elif high_low == 'high':
        plt.savefig('./latex/figures/multiscale_importance_graph_high_resolution.pdf', bbox_inches = 'tight')
    elif high_low == 'image':
        plt.savefig('.latex/figures/multiscale_importance_graph_image_patches.pdf', bbox_inches = 'tight')
    

plt.show(block=True)


# save image
fig = plt.figure(figsize = (4,2))
ax = fig.add_subplot(111)

imL = np.array(merged_dataset.load(example_num, which).imL_rgb)[100:380, 300:800]
ax.imshow(imL)
rect1 = patches.Rectangle((round(y_low - 40 - 300),round(x_low - 40 - 100)),80,80,linewidth=2,edgecolor='orange', facecolor = 'none')
rect2 = patches.Rectangle((round(y_high - 40 - 300),round(x_high - 40 - 100)),80,80,linewidth=2,edgecolor='yellow', facecolor = 'none')
ax.add_patch(rect1)
ax.add_patch(rect2)

# legend_elements = [Patch(facecolor='none', edgecolor='r', label='low resolution'),
#                    Patch(facecolor='none', edgecolor='c', label='high resolution')]

# ax.legend(handles = legend_elements)
plt.axis('off')
# plt.tight_layout()
if save_fig:
    plt.savefig('./latex/figures/multiscale_importance_image_patches_L.pdf', bbox_inches = 'tight')
plt.show(block=True)

# save image
fig = plt.figure(figsize = (4,2))
ax = fig.add_subplot(111)

imR = np.array(merged_dataset.load(example_num, which).imR_rgb)[100:380, 300:800]
ax.imshow(imR)
rect1 = patches.Rectangle((round(y_low - 40 - 300 - gt_low),round(x_low - 40 - 100)),80,80,linewidth=2,edgecolor='orange', facecolor = 'none')
rect2 = patches.Rectangle((round(y_high - 40 - 300 - gt_high),round(x_high - 40 - 100)),80,80,linewidth=2,edgecolor='yellow', facecolor = 'none')

ax.add_patch(rect1)
ax.add_patch(rect2)

# legend_elements = [Patch(facecolor='none', edgecolor='r', label='low resolution'),
#                    Patch(facecolor='none', edgecolor='c', label='high resolution')]

# ax.legend(handles = legend_elements)
plt.axis('off')
# plt.tight_layout()
if save_fig:
    plt.savefig('./latex/figures/multiscale_importance_image_patches_R.pdf', bbox_inches = 'tight')
plt.show(block=True)
