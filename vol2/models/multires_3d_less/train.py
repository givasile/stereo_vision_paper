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
import copy

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

action = 'finetune'
assert action in ['from_scratch', 'keep_training', 'finetune']

# directory to load_from
experiment_n_load_from = 1      # directory to load_from
experiment_n_save_to = 3        # directory to save_to
chekpoint_n = 10                 # which checkpoint to load weights/stats from
get_standart_dataset = True     # get_standart_dataset
which_dataset = 'kitti_2012'# which standard dataset to load from

# training parameters
train_for_epochs = 200            # how many epochs to train
lr = 0.001                      # learning rate
loss_from_scales_in_training = [2/3, 1/6, 1/6]

# where to validate on
train_on_crop = True            # training
val_on_val_crop = True         # validate on val_crop
val_on_val_full = True         # validate on val_full
val_on_test_crop = False         # validate on test_crop
val_on_test_full = False         # validate on test_full

device = 'cuda'                 # on which device to train

maxD = 192

####################################################
# exec init operations and define global variables #
####################################################

# create directory to save stats, weights, dataset
save_to = os.path.join(os.path.join(os.path.dirname(
    os.path.dirname(parent_path)), 'saved_models', 'vol2', cnn_name),
    'experiment_' + str(experiment_n_save_to))
if not os.path.exists(save_to):
    os.makedirs(save_to)

# create instance of model
if device == 'cpu':
    model_instance = net.model()
else:
    model_instance = net.model().cuda()

# init instance of optimizer
optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr,
                             betas=(0.9, 0.999))


if action is 'from_scratch':
    # load standard dataset or create
    if get_standart_dataset:
        parent = os.path.join(os.path.dirname(
            os.path.dirname(parent_path)), 'saved_models', 'common_datasets')
        dataset_path = os.path.join(parent, which_dataset + '.pickle')
        assert os.path.exists(dataset_path)
        with open(dataset_path, 'rb') as fm:
            dataset = pickle.load(fm)
    else:
        from_datasets = [{'kitti_2012': [0, 0, 0]},
                         {'kitti_2015': [0, 0, 0]},
                         {'freiburg_monkaa': [0, 0, 0]},
                         {'freiburg_driving': [0, 0, 0]},
                         {'freiburg_flying': [4, 8, 8]}]
        dataset = merged.dataset(from_datasets)
    with open(save_to + '/merged_dataset.pickle', 'wb') as fm:
        pickle.dump(dataset, fm)

    # initialize stats
    dic = {'_4':[], '_8': [], '_16': []}
    stats = {'train': {'loss': [], 'mae': copy.deepcopy(dic),
                       'loss_at_scales': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'val_crop': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'val_full': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'test_crop': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'test_full': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'general': {'cur_epoch': 1,
                         'cur_step': 1,
                         'val_on_val_crop': val_on_val_crop,
                         'val_on_val_full': val_on_val_full,
                         'val_on_test_crop': val_on_test_crop,
                         'val_on_test_full': val_on_test_full}}

    # init weights
    for m in model_instance.modules():
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.fill_(1)
        elif isinstance(m, torch.nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * \
                m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.fill_(1)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # save weights, optimizer, stats
    net.save_checkpoint(model_instance, optimizer, stats, save_to)

elif action == 'keep_training':
    parent = os.path.join(os.path.dirname(
        os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)

    # restore dataset
    dataset_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                                + '/merged_dataset.pickle')
    with open(dataset_path, 'rb') as fm:
        dataset = pickle.load(fm)

    # restore weights
    checkpoint_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                                   + '/checkpoint_' + str(chekpoint_n) + '.tar')
    checkpoint = torch.load(checkpoint_path)
    model_instance.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # restore stats
    stats = checkpoint['stats']


elif action == 'finetune':
    parent = os.path.join(os.path.dirname(
        os.path.dirname(parent_path)), 'saved_models/vol2', cnn_name)

    # init dataset
    if not get_standart_dataset:
        from_datasets = [{'kitti_2012': [0, 0, 0]},
                         {'kitti_2015': [0, 0, 0]},
                         {'freiburg_monkaa': [6931, 1733, 0]},
                         {'freiburg_driving': [3520, 880, 0]},
                         {'freiburg_flying': [17912, 4478, 4370]}]
        dataset = merged.dataset(from_datasets)
    else:
        parent1 = os.path.join(os.path.dirname(
            os.path.dirname(parent_path)), 'saved_models', 'common_datasets')
        dataset = os.path.join(parent1, which_dataset + '.pickle')
        with open(dataset, 'rb') as fm:
            dataset = pickle.load(fm)
    with open(save_to + '/merged_dataset.pickle', 'wb') as fm:
        pickle.dump(dataset, fm)

    # initialize stats
    dic = {'_4':[], '_8': [], '_16': []}
    stats = {'train': {'loss': [], 'mae': copy.deepcopy(dic),
                       'loss_at_scales': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'val_crop': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'val_full': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'test_crop': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'test_full': {'mae': copy.deepcopy(dic), 'pcg': copy.deepcopy(dic)},
             'general': {'cur_epoch': 1,
                         'cur_step': 1,
                         'val_on_val_crop': val_on_val_crop,
                         'val_on_val_full': val_on_val_full,
                         'val_on_test_crop': val_on_test_crop,
                         'val_on_test_full': val_on_test_full}}

    # restore weights
    checkpoint_path = os.path.join(parent, 'experiment_' + str(experiment_n_load_from)
                                   + '/checkpoint_' + str(chekpoint_n) + '.tar')
    checkpoint = torch.load(checkpoint_path)
    model_instance.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # save
    net.save_checkpoint(model_instance, optimizer, stats, save_to)


# number of trainable parameters
print('Cnn name: %s, number of parameters: %d' %(cnn_name, 
    sum([p.data.nelement() for p in model_instance.parameters()])))


start_full_time = time.time()
init_epoch = stats['general']['cur_epoch']

torch.cuda.empty_cache()
# START EPOCH
for ep in range(init_epoch, init_epoch + train_for_epochs):

    # TRAINING PART
    if train_on_crop:
        net.training_epoch(dataset, 1, stats, model_instance, optimizer, maxD,
                           loss_from_scales_in_training, device)
    net.save_checkpoint(model_instance, optimizer, stats, save_to)
    torch.cuda.empty_cache()
    
    # VAL_CROP
    if val_on_val_crop:
        net.validate('val', 'crop', dataset, 2, stats, model_instance, device, maxD)
    net.save_checkpoint(model_instance, optimizer, stats, save_to)
    torch.cuda.empty_cache()
    
    # VAL_FULL
    if val_on_val_full:
        net.validate('val', 'full', dataset, 1, stats, model_instance, device, maxD)
    net.save_checkpoint(model_instance, optimizer, stats, save_to)
    torch.cuda.empty_cache()

    # TEST_CROP
    if val_on_test_crop:
        net.validate('test', 'crop', dataset, 4, stats, model_instance, device, maxD)
    net.save_checkpoint(model_instance, optimizer, stats, save_to)
    torch.cuda.empty_cache()
    
    # TEST_FULL
    if val_on_test_full:
        net.validate('test', 'full', dataset, 2, stats, model_instance, device, maxD)
    net.save_checkpoint(model_instance, optimizer, stats, save_to)
    torch.cuda.empty_cache()
    
    # update current epoch
    stats['general']['cur_epoch'] += 1

    # save checkpoint
    net.save_checkpoint(model_instance, optimizer, stats, save_to,
                        stats['general']['cur_epoch'] - 1)

    # empty cuda
    torch.cuda.empty_cache()

print('full training time = %.2f hours' %
      ((time.time() - start_full_time)/3600))
