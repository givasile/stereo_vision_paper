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

# np.random.seed(1)
# random.seed(1)
cnn_name = 'merging_info_net_custom_features_free_2d_3d_weights'

# import config file
conf_path = './../conf.ini'
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

# directory to load_from
experiment_n_load_from = 1      # directory to load_from
checkpoint_n = 10                # which checkpoint to load weights/stats from

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
example_num = 150

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
scales = [[round(max_disp/4), round(h/4), round(w/4)],
          [round(max_disp/8), round(h/8), round(w/8)],
          [round(max_disp/16), round(h/16), round(w/16)],
          [round(max_disp/32), round(h/32), round(w/32)]]

prediction_from_scales = {3: ['after'],
                          2: ['after'],
                          1: ['after'],
                          0: ['after']}


tmp = timeit.default_timer()
mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)
print("Inspection execution time: %s" % (timeit.default_timer()-tmp))


# visualize
imL = dataset.load(example_num, which).imL_rgb
imR = dataset.load(example_num, which).imR_rgb

imL_0 = imL.resize((scales[0][2], scales[0][1]), PIL.Image.BILINEAR)
imR_0 = imR.resize((scales[0][2], scales[0][1]), PIL.Image.BILINEAR)

imL_1 = imL.resize((scales[1][2], scales[1][1]), PIL.Image.BILINEAR)
imR_1 = imR.resize((scales[1][2], scales[1][1]), PIL.Image.BILINEAR)

imL_2 = imL.resize((scales[2][2], scales[2][1]), PIL.Image.BILINEAR)
imR_2 = imR.resize((scales[2][2], scales[2][1]), PIL.Image.BILINEAR)

imL_3 = imL.resize((scales[3][2], scales[3][1]), PIL.Image.BILINEAR)
imR_3 = imR.resize((scales[3][2], scales[3][1]), PIL.Image.BILINEAR)


pred_0 = predictions_dict[0]['after'].cpu().numpy()[0]
# save
base_path = "/home/givasile/stereo_vision/paper/images/figure_1/"
with open(base_path + "block_diagram_imL.png", 'wb') as fm:
    imL.save(fm)
with open(base_path + "block_diagram_imR.png", 'wb') as fm:
    imR.save(fm)
with open(base_path + "block_diagram_imL_0.png", 'wb') as fm:
    imL_0.save(fm)
with open(base_path + "block_diagram_imR_0.png", 'wb') as fm:
    imR_0.save(fm)
with open(base_path + "block_diagram_imL_1.png", 'wb') as fm:
    imL_1.save(fm)
with open(base_path + "block_diagram_imR_1.png", 'wb') as fm:
    imR_1.save(fm)
with open(base_path + "block_diagram_imL_2.png", 'wb') as fm:
    imL_2.save(fm)
with open(base_path + "block_diagram_imR_2.png", 'wb') as fm:
    imR_2.save(fm)
with open(base_path + "block_diagram_imL_3.png", 'wb') as fm:
    imL_3.save(fm)
with open(base_path + "block_diagram_imR_3.png", 'wb') as fm:
    imR_3.save(fm)

# save_numpy_image(
#     predictions_dict[0]['after'][0],
#     "/home/givasile/stereo_vision/paper/block_diagram_pred_0.png",
#     [0, dispL.max()]
#     )


# def save_numpy_image(im, path, limits):
#     fig = plt.figure(figsize = (1,1), frameon = False)
#     fig.set_size_inches(im.shape)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     tt = ax.imshow(im)
#     tt.set_clim(limits)
#     plt.show()
    # fig.savefig(path, bbox_inches='tight', pad_inches=0)
    
# imL = imL.squeeze(0).cpu().numpy()

# visualize.imshow_3ch(imL.squeeze(0).cpu(), 'imL')
# visualize.imshow_3ch(imR.squeeze(0).cpu(), 'imR')

# for i, im in enumerate(predictions_dict.values()):
#     time.sleep(1)
#     visualize.imshow_1ch(im['after'][0].cpu(), str(i), [0, dispL.max()])
# visualize.imshow_1ch(dispL[0].cpu(), 'dispL')
# image_pcg = evaluate.image_percentage_over_limit(predictions_dict[0]['after'].cpu(),
#                                                  dispL.cpu(), maskL.cpu(), 3)
# visualize.imshow_1ch(image_pcg[0], 'over_thres')
