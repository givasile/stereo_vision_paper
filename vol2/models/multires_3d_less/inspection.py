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
cnn_name = 'multires_3d_less'

# import config file
conf_path = './../../../conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

# import custom modules
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

# which image to inspect
which = 'test'                 # ['train', 'val', 'test']
form = 'full_im'                   # ['crop', 'full_im']
limit_maxD = True               # extract statistics only on disp_gt < maxD
get_standart_dataset = True    # image from standard dataset or the intrinsic
which_dataset = 'freiburg_tr_te'    # iff standard dataset, the name of it
example_num = 0                 # index of example
device = 'cuda'                 # device to run on
maxD = 192                      # maximum disparity under computation
mode = 'eval'                   # ['train', 'eval']

assert which_dataset in ['flying_tr_te',
                         'freiburg',
                         'freiburg_tr_te',
                         'kitti_2012',
                         'kitti_2015']

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
    dataset_path = os.path.join(parent, which_dataset + '.pickle')
    assert os.path.exists(dataset_path)
    with open(dataset_path, 'rb') as fm:
        dataset = pickle.load(fm)
else:
    parent = os.path.join(os.path.dirname(
        os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)
    dataset_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                                + '/merged_dataset.pickle')
    with open(dataset_path, 'rb') as fm:
        dataset = pickle.load(fm)


# restore weights
parent = os.path.join(os.path.dirname(
    os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)
checkpoint_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                               + '/checkpoint_' + str(checkpoint_n) + '.tar')
checkpoint = torch.load(checkpoint_path)
model_instance.load_state_dict(checkpoint['state_dict'])

# restore stats
stats = checkpoint['stats']

# print number of parameters
print('Cnn name: %s, number of parameters: %d' %(cnn_name, 
    sum([p.data.nelement() for p in model_instance.parameters()])))


# load examples
data_feeder = preprocess.dataset(dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0)
imR = imR.unsqueeze(0)
max_limit = dispL.max()
dispL = dispL.unsqueeze(0)
maskL = maskL.unsqueeze(0)

# forward pass
start_time = time.time()
if device == 'cuda':
    imL = imL.cuda()
    imR = imR.cuda()
    dispL = dispL.cuda()
    maskL = maskL.cuda()
    
tmp = net.inspection(model_instance, device, mode, imL, imR, dispL, maskL, maxD)

if tmp is not None:
    mae, pcg, imL_d, imR_d, pred_4, pred_8, pred_16, imL_d, imR_d, cost_a, cost_b_4, cost_b_8, cost_b_16 = tmp

end_time = time.time()
elapsed_time = end_time - start_time

# error
threshold = 5
err_im = evaluate.image_absolute_error(pred_4, dispL, maskL)
over_thres_im = evaluate.image_percentage_over_limit(pred_4, dispL, maskL, threshold)
print(' Time: %.3f \n mean error: %.3f px \n percentage over %d pixel: %f'
      % (elapsed_time, mae[0], threshold, pcg[0]))

# visualize
visualize.imshow_mch(imL[0].cpu(), 0, 'imL')
visualize.imshow_mch(imR[0].cpu(), 0, 'imR')
visualize.imshow_1ch(pred_4[0].cpu(), 'prediction', [
                     dispL.min().item(), dispL.max().item()])
visualize.imshow_1ch(pred_8[0].cpu(), 'prediction', [
                     dispL.min().item(), dispL.max().item()])
visualize.imshow_1ch(pred_16[0].cpu(), 'prediction', [
                     dispL.min().item(), dispL.max().item()])
visualize.imshow_1ch(dispL[0].cpu(), 'ground truth')
visualize.imshow_1ch(over_thres_im[0].cpu(), 'error over threshold')
