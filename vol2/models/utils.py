import raw_dataset.merged_dataset as merged_dataset
import torch
import math
import configparser
import os
import pickle


def load_common_dataset(dataset_name: str, common_dataset_dir: str) -> merged_dataset.Dataset:
    with open(os.path.join(common_dataset_dir, dataset_name + '.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)


def load_specific_dataset(dataset_dir: str) -> merged_dataset.Dataset:
    print("Loading specific dataset... \n")
    with open(os.path.join(dataset_dir, 'merged_dataset_state_dict.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)


def create_directory_to_store_experiment(conf: configparser.ConfigParser, cnn_name: str, experiment_n: int) -> str:
    save_dir = os.path.join(conf['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n))
    if not os.path.exists(save_dir):
        print("Creating directory for storing the experiment... ")
        os.makedirs(save_dir)
    return save_dir


def store_state_dict_of_dataset(dataset: merged_dataset.Dataset, dir_to_save: str):
        split_dat = ['freiburg_flying_split', 'freiburg_monkaa_split', 'freiburg_driving_split', 'kitti_2012_split',
                     'kitti_2015_split']
        dic = {}
        for dat_name in split_dat:
            dat_inst = getattr(dataset, dat_name)
            dic[dat_name] = {}
            dic[dat_name]['split'] = dat_inst.split
            dic[dat_name]['train'] = dat_inst.train
            dic[dat_name]['val'] = dat_inst.val
            dic[dat_name]['test'] = dat_inst.test

        print("Saving state dict of dataset... \n")
        with open(os.path.join(dir_to_save, 'merged_dataset_state_dict.pickle'), 'wb') as fm:
            pickle.dump(dic, fm)


def initialize_stats_dict_for_merging_info_net(val_on_val_crop, val_on_val_full, val_on_test_crop, val_on_test_full):
    # initialize stats
    def gen_dict(prediction_from_scales):
        tmp = None
        aa = isinstance(prediction_from_scales[next(
            iter(prediction_from_scales))], list)
        if aa:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key]:
                    if tmp is None:
                        tmp = {str(key) + '_' + key1: []}
                    else:
                        tmp[str(key) + '_' + key1] = []
        else:
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key].keys():
                    if tmp is None:
                        tmp = {str(key) + '_' + key1: []}
                    else:
                        tmp[str(key) + '_' + key1] = []
        return tmp

    prediction_from_scales = {3: ['after'],
                              2: ['after'],
                              1: ['after'],
                              0: ['after']}
    loss_from_scales_in_training = {3: {'after': 1 / 4},
                                    2: {'after': 1 / 4},
                                    1: {'after': 1 / 4},
                                    0: {'after': 1 / 4}}
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
    return stats


def initialize_weights_of_cnn(model_instance):
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

