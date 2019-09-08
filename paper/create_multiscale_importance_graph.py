'''
Script for figure 3 of the paper, where it is proven with a real case scenario 
(2 specific points are choosen) the importance of multiscale analysis.

'''
import sys
import configparser
import importlib
import torch
import timeit
import time
import pickle
import os
import math
import numpy as np
import random
from scipy.special import softmax
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


save_fig = True

# np.random.seed(1)
# random.seed(1)
cnn_name = 'merging_info_net_custom_features'

# import config file
conf_path = '../conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

# import custom modulesn
submodules = importlib.import_module('vol2.models.' + cnn_name + '.submodules')
net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
merged = importlib.import_module('raw_dataset.merged_dataset')
preprocess = importlib.import_module('preprocess')
evaluate = importlib.import_module('evaluate')
visualize = importlib.import_module('visualize')

####################################################
################# Configuration ####################
####################################################
high_low = 'high'

# directory to load_from
experiment_n_load_from = 1      # directory to load_from
checkpoint_n = 15                # which checkpoint to load weights/stats from

# preproccess
which = 'test'
form = 'full_im'
limit_maxD = True
get_standart_dataset = True
which_dataset = 'freiburg_tr_te'
assert which_dataset in ['flying_tr_te',
                         'freiburg',
                         'freiburg_tr_te',
                         'kitti_2012',
                         'kitti_2015']
# example_num
example_num = 100

device = 'cuda'

####################################################
# exec init operations and define global variables #
####################################################

# create instance of model
if device == 'cpu':
    model_instance = net.model()
else:
    model_instance = net.model().cuda()

# restore dataset
if get_standart_dataset:
    parent = os.path.join(os.path.dirname(
        os.path.dirname(parent_path)), 'saved_models', 'common_datasets')
    dataset = os.path.join(parent, which_dataset + '.pickle')
    assert os.path.exists(dataset)
    with open(dataset, 'rb') as fm:
        dataset = pickle.load(fm)
else:
    parent = os.path.join(os.path.dirname(
    os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)

    dataset_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                                + '/merged_dataset.pickle')
    with open(dataset_path, 'rb') as fm:
        dataset = pickle.load(fm)

parent = os.path.join(os.path.dirname(
    os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)

# restore weights
checkpoint_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                               + '/checkpoint_' + str(checkpoint_n) + '.tar')
checkpoint = torch.load(checkpoint_path)
model_instance.load_state_dict(checkpoint['state_dict'])

# restore stats
stats = checkpoint['stats']

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model_instance.parameters()])))

# input
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

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

# visualize.imshow_3ch(imL.squeeze(0).cpu(), 'imL')
# visualize.imshow_3ch(imR.squeeze(0).cpu(), 'imR')

# for i, im in enumerate(predictions_dict.values()):
#     time.sleep(1)
#     visualize.imshow_1ch(im['after'][0].cpu(), str(i), [0, dispL.max()])
# visualize.imshow_1ch(dispL[0].cpu(), 'dispL')
# image_pcg = evaluate.image_percentage_over_limit(predictions_dict[0]['after'].cpu(),
#                                                  dispL.cpu(), maskL.cpu(), 3)
# visualize.imshow_1ch(image_pcg[0], 'over_thres')


# multiscale prediction 
scales = [[round(max_disp/4), round(h/4), round(w/4)],
          [round(max_disp/8), round(h/8), round(w/8)],
          [round(max_disp/16), round(h/16), round(w/16)],
          [round(max_disp/32), round(h/32), round(w/32)]]
prediction_from_scales = {0: ['after'],
                          1: ['after'],
                          2: ['after'],
                          3: ['after']}
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

# example where law resolution matters
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

ax.set_ylabel('scale', fontdict = font, labelpad = -15)
ax.set_yticks([])

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

ax.set_ylabel('scale', fontdict = font, labelpad = -15)
ax.set_yticks([])

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
        plt.savefig('/home/givasile/stereo_vision/paper/multiscale_importance/multiscale_importance_graph_low_resolution.png', bbox_inches = 'tight')
    elif high_low == 'high':
        plt.savefig('/home/givasile/stereo_vision/paper/multiscale_importance/multiscale_importance_graph_high_resolution.png', bbox_inches = 'tight')
    elif high_low == 'image':
        plt.savefig('/home/givasile/stereo_vision/paper/multiscale_importance/multiscale_importance_graph_image_patches.png', bbox_inches = 'tight')
    

plt.show(block=False)


# save image
fig = plt.figure(figsize = (4,2))
ax = fig.add_subplot(111)

imL = np.array(dataset.load(example_num, which).imL_rgb)[100:380, 300:800]
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
plt.show(block=False)
if save_fig:
    plt.savefig('/home/givasile/stereo_vision/paper/multiscale_importance/multiscale_importance_image_patches.png', bbox_inches = 'tight')
