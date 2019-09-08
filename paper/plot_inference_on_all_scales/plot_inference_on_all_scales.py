import sys
import PIL
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
import matplotlib.pyplot as plt

# np.random.seed(1)
# random.seed(1)
cnn_name = 'merging_info_net_custom_features'

# import config file
conf_path = './../../conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

# import custom modulesn
# submodules = importlib.import_module('vol2.models.' + cnn_name + '.submodules')
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
def plot_and_save_PIL(im, name):
    base_path = "/home/givasile/stereo_vision/paper/images/figure_2/"
    with open(base_path + name + '.png', 'wb') as fm:
        im.save(fm)

def plot_and_save(im, name):
    base_path = '/home/givasile/stereo_vision/paper/images/figure_2/'
    plt.figure();
    # plt.title(name);
    im = plt.imshow(im);
    plt.axis('off');
    im.set_cmap('gray');
    plt.show(block=False)
    plt.savefig(base_path + name + '.png', bbox_inches = 'tight')
    
def plot_err_image(image, name):
    base_path = '/home/givasile/stereo_vision/paper/images/figure_2/'
    image = image[0].numpy()
    image[image > 5] = 5
    plt.figure();
    # plt.title(name);
    plt.axis("off")
    im = plt.imshow(image);
    # plt.colorbar()
    im.set_cmap('afmhot');
    plt.show(block=False)
    plt.savefig(base_path + name + '.png', bbox = 'tight')

    
imL = dataset.load(example_num, which).imL_rgb
plot_and_save_PIL(imL, 'imL')

imL_0 = imL.resize((scales[0][2], scales[0][1]), PIL.Image.BILINEAR)
plot_and_save_PIL(imL_0, 'imL_0')

imL_1 = imL.resize((scales[1][2], scales[1][1]), PIL.Image.BILINEAR)
plot_and_save_PIL(imL_1, 'imL_1')

imL_2 = imL.resize((scales[2][2], scales[2][1]), PIL.Image.BILINEAR)
plot_and_save_PIL(imL_2, 'imL_2')

imL_3 = imL.resize((scales[3][2], scales[3][1]), PIL.Image.BILINEAR)
plot_and_save_PIL(imL_3, 'imL_3')

pred_comb_0 = predictions_dict[0]['after'].cpu().numpy()[0]
pred_comb_0[pred_comb_0 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_comb_0_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_comb_0, 0)),
                                                dispL.cpu(), maskL.cpu())
plot_err_image(pred_comb_0_err, 'pred_comb_0_err')
plot_and_save(pred_comb_0, 'pred_comb_0')

pred_comb_1 = predictions_dict[1]['after'].cpu().numpy()[0]
pred_comb_1[pred_comb_1 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_comb_1_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_comb_1, 0)),
                                                dispL.cpu(), maskL.cpu())
plot_err_image(pred_comb_1_err, 'pred_comb_1_err')
plot_and_save(pred_comb_1, 'pred_comb_1')

pred_comb_2 = predictions_dict[2]['after'].cpu().numpy()[0]
pred_comb_2[pred_comb_2 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_comb_2_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_comb_2, 0)),
                                                dispL.cpu(), maskL.cpu())
plot_err_image(pred_comb_2_err, 'pred_comb_2_err')
plot_and_save(pred_comb_2, 'pred_comb_2')

pred_comb_3 = predictions_dict[3]['after'].cpu().numpy()[0]
pred_comb_3[pred_comb_3 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_comb_3_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_comb_3, 0)),
                                                dispL.cpu(), maskL.cpu())
plot_err_image(pred_comb_3_err, 'pred_comb_3_err')
plot_and_save(pred_comb_3, 'pred_comb_3')


# only at scale 0
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/4), round(h/4), round(w/4)]]

prediction_from_scales = {0: ['after']}

mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)

pred_0 = predictions_dict[0]['after'].cpu().numpy()[0]
pred_0[pred_0 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_0_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_0, 0)),
                                           dispL.cpu(), maskL.cpu())
plot_err_image(pred_0_err, 'pred_0_err')
plot_and_save(pred_0, 'pred_0')


# only at scale 1
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/8), round(h/8), round(w/8)]]

prediction_from_scales = {0: ['after']}

mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)

pred_1 = predictions_dict[0]['after'].cpu().numpy()[0]
pred_1[pred_1 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_1_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_1, 0)),
                                           dispL.cpu(), maskL.cpu())
plot_err_image(pred_1_err, 'pred_1_err')
plot_and_save(pred_1, 'pred_1')

# only at scale 2
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/16), round(h/16), round(w/16)]]

prediction_from_scales = {0: ['after']}

mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)

pred_2 = predictions_dict[0]['after'].cpu().numpy()[0]
pred_2[pred_2 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_2_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_2, 0)),
                                           dispL.cpu(), maskL.cpu())
plot_err_image(pred_2_err, 'pred_2_err')
plot_and_save(pred_2, 'pred_2')

# only at scale 3
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/32), round(h/32), round(w/32)]]

prediction_from_scales = {0: ['after']}

mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)

pred_3 = predictions_dict[0]['after'].cpu().numpy()[0]
pred_3[pred_3 > dispL.max().cpu().item()] = dispL.max().cpu().item()
pred_3_err = evaluate.image_absolute_error(torch.tensor(np.expand_dims(pred_3, 0)),
                                           dispL.cpu(), maskL.cpu())
plot_and_save(pred_3, 'pred_3')
plot_err_image(pred_3_err, 'pred_3_err')
