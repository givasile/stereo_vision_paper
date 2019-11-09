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
conf_path = './../conf.ini'
conf_file = configparser.ConfigParser()
conf_file.read(conf_path)

# add parent path, if not already added
parent_path = conf_file['PATHS']['PARENT_DIR']
sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

# import all needed modules
utils = importlib.import_module('vol2.models.utils')
net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
preprocess = importlib.import_module('preprocess')
evaluate = importlib.import_module('evaluate')

experiment_n = 1      # directory to load_from
checkpoint_n = 15                # which checkpoint to load weights/stats from
which = 'test'
form = 'full_im'
limit_maxD = True
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 100
device = 'cuda'

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
if device == 'cpu':
    model_instance = net.model()
else:
    model_instance = net.model().cuda()

# restore weights
checkpoint = torch.load(checkpoint_filepath)
model_instance.load_state_dict(checkpoint['state_dict'])

# restore training statistics
stats = checkpoint['stats']

# input
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
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
    base_path = "./latex/figures/"
    with open(base_path + name + '.png', 'wb') as fm:
        im.save(fm)

def plot_and_save(im, name):
    base_path = './latex/figures/'
    plt.figure();
    # plt.title(name);
    im = plt.imshow(im);
    plt.axis('off');
    im.set_cmap('gray');
    plt.savefig(base_path + name + '.png', bbox_inches = 'tight')
    plt.show(block=False)

    
def plot_err_image(image, name):
    base_path = './latex/figures/'
    image = image[0].numpy()
    image[image > 5] = 5
    plt.figure();
    # plt.title(name);
    plt.axis("off")
    im = plt.imshow(image);
    # plt.colorbar()
    im.set_cmap('afmhot');
    plt.savefig(base_path + name + '.png', bbox = 'tight')
    plt.show(block=False)


imL = merged_dataset.load(example_num, which).imL_rgb
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
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
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
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
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
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
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
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
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
