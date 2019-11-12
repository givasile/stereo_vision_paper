import sys
import configparser
import importlib
import torch
import time
import os


cnn_name = 'merging_info_net_custom_features'

action = 'finetune'
assert action in ['from_scratch', 'keep_training', 'finetune']

# directory to load_from
experiment_n_load_from = 1  # directory to load_from
experiment_n_save_to = 103  # directory to save_to
chekpoint_n = 15  # which checkpoint to load weights/stats from
get_common_dataset = True  # get_standart_dataset
common_dataset_name = 'flying_tr_te'  # which standard dataset to load from

# training parameters
train_for_epochs = 1  # how many epochs to train
lr = 0.001  # learning rate

# where to validate on
train_on_crop = False  # training
val_on_val_crop = False  # validate on val_crop
val_on_val_full = False  # validate on val_full
val_on_test_crop = False  # validate on test_crop
val_on_test_full = True  # validate on test_full

device = 'cuda'  # on which device to train

dataset_mixture = {'kitti_2012': [0, 0, 0],
                 'kitti_2015': [0, 0, 0],
                 'freiburg_monkaa': [1, 1, 0],
                 'freiburg_driving': [1, 1, 0],
                 'freiburg_flying': [2, 2, 2]}


def run(conf_file):
    # import custom modules
    net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
    merged = importlib.import_module('raw_dataset.merged_dataset')
    utils = importlib.import_module('vol2.models.utils')

    # create directory to save stats, weights, dataset
    experiment_directory = utils.create_directory_to_store_experiment(conf_file, cnn_name, experiment_n_save_to)
    directory_to_load_from = os.path.join(conf['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n_load_from))

    # create instance of model
    if device == 'cpu':
        model_instance = net.model()
    else:
        model_instance = net.model().cuda()

    # init instance of optimizer
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr, betas=(0.9, 0.999))

    # get dataset
    if action in ['from_scratch', 'finetune']:
        if get_common_dataset:
            dataset = utils.load_common_dataset(common_dataset_name, conf_file['PATHS']['common_datasets'])
        else:
            dataset = merged.Dataset(dataset_mixture, create_load='create', state_dict=None)
        # save dataset to directory
        utils.store_state_dict_of_dataset(dataset, experiment_directory)
    elif action == 'keep_training':
        dataset = utils.load_specific_dataset(experiment_directory)

    # get stats
    if action in ['from_scratch', 'finetune']:
        stats = utils.initialize_stats_dict_for_merging_info_net(val_on_val_crop, val_on_val_full, val_on_test_crop, val_on_test_full)
    elif action == 'keep_training':
        checkpoint_filepath = os.path.join(directory_to_load_from, 'checkpoint_' + str(chekpoint_n) + '.tar')
        checkpoint = torch.load(checkpoint_filepath)
        stats = checkpoint['stats']


    # get or init weights
    if action in ['from_scratch']:
        utils.initialize_weights_of_cnn(model_instance)
    elif action == 'keep_training':
        model_instance.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif action == 'finetune':
        checkpoint_filepath = os.path.join(directory_to_load_from, 'checkpoint_' + str(chekpoint_n) + '.tar')
        checkpoint = torch.load(checkpoint_filepath)
        model_instance.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # save checkpoint
    if action in ['from_scratch', 'finetune']:
        net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_instance.parameters()])))


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

            net.training_epoch(dataset, 2, stats, model_instance, optimizer, initial_scale,
                               scales, prediction_from_scales, loss_from_scales_in_training, device)
            net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)
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
            net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)
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
            net.validate('val', 'full', dataset, 1, stats, model_instance, initial_scale,
                         scales, prediction_from_scales, device)
            net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)
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

            net.validate('test', 'crop', dataset, 4, stats, model_instance, initial_scale,
                         scales, prediction_from_scales, device)
            net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)
            torch.cuda.empty_cache()

        # TEST_FULL
        if val_on_test_full:
            max_disp = 192
            h = 544
            w = 960
            initial_scale = [max_disp, h, w]
            scales = [[round(max_disp/16), round(h/16), round(w/16)]]
            prediction_from_scales = {0: ['after']}

            net.validate('test', 'full', dataset, 1, stats, model_instance, initial_scale,
                         scales, prediction_from_scales, device)
            net.save_checkpoint(model_instance, optimizer, stats, experiment_directory)
            torch.cuda.empty_cache()

        # update current epoch
        stats['general']['cur_epoch'] += 1

        # save checkpoint
        net.save_checkpoint(model_instance, optimizer, stats, experiment_directory,
                            stats['general']['cur_epoch'] - 1)

        # empty cuda
        torch.cuda.empty_cache()

    print('Full training time = %.2f hours' %
          ((time.time() - start_full_time)/3600))


if __name__ == '__main__':
    # import config file
    conf_path = './../../conf.ini'
    conf = configparser.ConfigParser()
    conf.read(conf_path)

    # add parent path, if not already added
    parent_path = conf['PATHS']['parent_dir']
    sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

    # run main
    run(conf)
