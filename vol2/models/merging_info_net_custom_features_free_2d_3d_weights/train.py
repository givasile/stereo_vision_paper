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

action = 'from_scratch'
assert action in ['from_scratch', 'keep_training', 'finetune']

# directory to load_from
experiment_n_load_from = 1      # directory to load_from
experiment_n_save_to = 25        # directory to save_to
chekpoint_n = 13                 # which checkpoint to load weights/stats from
get_standart_dataset = True     # get_standart_dataset
which_dataset = 'flying_tr_te_50'  # which standard dataset to load from

# training parameters
train_for_epochs = 10           # how many epochs to train
lr = 0.001                      # learning rate

# where to validate on
train_on_crop = True            # training
val_on_val_crop = False         # validate on val_crop
val_on_val_full = False        # validate on val_full
val_on_test_crop = True         # validate on test_crop
val_on_test_full = True         # validate on test_full

device = 'cuda'                 # on which device to train

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

# init dataset, weights, statistics
if action is 'from_scratch':
    # load standard dataset or create
    if get_standart_dataset:
        parent = os.path.join(os.path.dirname(
            os.path.dirname(parent_path)), 'saved_models', 'common_datasets')
        dataset = os.path.join(parent, which_dataset + '.pickle')
        assert os.path.exists(dataset)
        with open(dataset, 'rb') as fm:
            dataset = pickle.load(fm)
    else:
        from_datasets = [{'kitti_2012': [0, 0, 0]},
                         {'kitti_2015': [0, 0, 0]},
                         {'freiburg_monkaa': [0, 0, 0]},
                         {'freiburg_driving': [0, 1, 0]},
                         {'freiburg_flying': [22390, 0, 4370]}]
        dataset = merged.dataset(from_datasets)
    with open(save_to + '/merged_dataset.pickle', 'wb') as fm:
        pickle.dump(dataset, fm)

    # initialize stats
    def gen_dict(prediction_from_scales):
        tmp = None
        aa = isinstance(prediction_from_scales[next(
            iter(prediction_from_scales))], list)
        if aa:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key]:
                    if tmp is None:
                        tmp = {str(key)+'_'+key1: []}
                    else:
                        tmp[str(key)+'_'+key1] = []
        else:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key].keys():
                    if tmp is None:
                        tmp = {str(key)+'_'+key1: []}
                    else:
                        tmp[str(key)+'_'+key1] = []
        return tmp

    prediction_from_scales = {3: ['after'],
                              2: ['after'],
                              1: ['after'],
                              0: ['after']}
    loss_from_scales_in_training = {3: {'after': 1/4},
                                    2: {'after': 1/4},
                                    1: {'after': 1/4},
                                    0: {'after': 1/4}}
    tmp = prediction_from_scales
    tmp1 = loss_from_scales_in_training
    stats = {'train': {'mae': gen_dict(tmp1), 'pcg': gen_dict(tmp1), 'loss': []},
             'val_crop': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'val_full': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'test_crop': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'test_full': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
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
    def gen_dict(prediction_from_scales):
        tmp = None
        aa = isinstance(prediction_from_scales[next(
            iter(prediction_from_scales))], list)
        if aa:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key]:
                    if tmp is None:
                        tmp = {str(key)+'_'+key1: []}
                    else:
                        tmp[str(key)+'_'+key1] = []
        else:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key].keys():
                    if tmp is None:
                        tmp = {str(key)+'_'+key1: []}
                    else:
                        tmp[str(key)+'_'+key1] = []
        return tmp

    prediction_from_scales = {3: ['after'],
                              2: ['after'],
                              1: ['after'],
                              0: ['after']}
    loss_from_scales_in_training = {3: {'after': 1/4},
                                    2: {'after': 1/4},
                                    1: {'after': 1/4},
                                    0: {'after': 1/4}}
    tmp = prediction_from_scales
    tmp1 = loss_from_scales_in_training
    stats = {'train': {'mae': gen_dict(tmp1), 'pcg': gen_dict(tmp1), 'loss': []},
             'val_crop': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'val_full': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'test_crop': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
             'test_full': {'mae': gen_dict(tmp), 'pcg': gen_dict(tmp)},
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


print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model_instance.parameters()])))


# training for n epochs
start_full_time = time.time()
init_epoch = stats['general']['cur_epoch']
torch.cuda.empty_cache()
for ep in range(init_epoch, init_epoch + train_for_epochs):

    # TRAINING PART
    if train_on_crop:
        max_disp = 192
        h = 256
        w = 512
        initial_scale = [max_disp, h, w]
        scales = [[round(max_disp/4), round(h/4), round(w/4)],
                  [round(max_disp/8), round(h/8), round(w/8)],
                  [round(max_disp/16), round(h/16), round(w/16)],
                  [round(max_disp/32), round(h/32), round(w/32)]]
        prediction_from_scales = {3: ['after'],
                                  2: ['after'],
                                  1: ['after'],
                                  0: ['after']}
        loss_from_scales_in_training = {3: {'after': 1/4/2},
                                        2: {'after': 1/4/2},
                                        1: {'after': 1/4/2},
                                        0: {'after': 1/2}}

        net.training_epoch(dataset, 1, stats, model_instance, optimizer, initial_scale,
                           scales, prediction_from_scales, loss_from_scales_in_training, device)
        net.save_checkpoint(model_instance, optimizer, stats, save_to)
        torch.cuda.empty_cache()

    # VAL_CROP
    if val_on_val_crop:
        max_disp = 192
        h = 256
        w = 768
        initial_scale = [max_disp, h, w]
        scales = [[round(max_disp/4), round(h/4), round(w/4)],
                  [round(max_disp/8), round(h/8), round(w/8)],
                  [round(max_disp/16), round(h/16), round(w/16)],
                  [round(max_disp/32), round(h/32), round(w/32)]]
        prediction_from_scales = {3: ['after'],
                                  2: ['after'],
                                  1: ['after'],
                                  0: ['after']}
        net.validate('val', 'crop', dataset, 4, stats, model_instance, initial_scale,
                     scales, prediction_from_scales, device)
        net.save_checkpoint(model_instance, optimizer, stats, save_to)
        torch.cuda.empty_cache()

    # VAL_FULL
    if val_on_val_full:
        max_disp = 192
        h = 368
        w = 1232
        initial_scale = [max_disp, h, w]
        scales = [[round(max_disp/4), round(h/4), round(w/4)],
                  [round(max_disp/8), round(h/8), round(w/8)],
                  [round(max_disp/16), round(h/16), round(w/16)],
                  [round(max_disp/32), round(h/32), round(w/32)]]
        prediction_from_scales = {3: ['after'],
                                  2: ['after'],
                                  1: ['after'],
                                  0: ['after']}
        net.validate('val', 'full', dataset, 2, stats, model_instance, initial_scale,
                     scales, prediction_from_scales, device)
        net.save_checkpoint(model_instance, optimizer, stats, save_to)
        torch.cuda.empty_cache()

    # TEST_CROP
    if val_on_test_crop:
        max_disp = 192
        h = 256
        w = 512
        initial_scale = [max_disp, h, w]
        scales = [[round(max_disp/4), round(h/4), round(w/4)],
                  [round(max_disp/8), round(h/8), round(w/8)],
                  [round(max_disp/16), round(h/16), round(w/16)],
                  [round(max_disp/32), round(h/32), round(w/32)]]
        prediction_from_scales = {3: ['after'],
                                  2: ['after'],
                                  1: ['after'],
                                  0: ['after']}

        net.validate('test', 'crop', dataset, 2, stats, model_instance, initial_scale,
                     scales, prediction_from_scales, device)
        net.save_checkpoint(model_instance, optimizer, stats, save_to)
        torch.cuda.empty_cache()

    # TEST_FULL
    if val_on_test_full:
        max_disp = 192
        h = 544
        w = 960
        initial_scale = [max_disp, h, w]
        scales = [[round(max_disp/4), round(h/4), round(w/4)],
                  [round(max_disp/8), round(h/8), round(w/8)],
                  [round(max_disp/16), round(h/16), round(w/16)],
                  [round(max_disp/32), round(h/32), round(w/32)]]
        prediction_from_scales = {3: ['after'],
                                  2: ['after'],
                                  1: ['after'],
                                  0: ['after']}

        net.validate('test', 'full', dataset, 2, stats, model_instance, initial_scale,
                     scales, prediction_from_scales, device)
        net.save_checkpoint(model_instance, optimizer, stats, save_to)
        torch.cuda.empty_cache()

    # update current epoch
    stats['general']['cur_epoch'] += 1

    # save checkpoint
    net.save_checkpoint(model_instance, optimizer, stats, save_to,
                        stats['general']['cur_epoch'] - 1)

    # empty cuda
    torch.cuda.empty_cache()

print('Full training time = %.2f hours' %
      ((time.time() - start_full_time)/3600))
