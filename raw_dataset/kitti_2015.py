import os
import pickle
import raw_dataset.utils as utils
import raw_dataset.abstract_classes as abc
import raw_dataset.kitti_utils as kitti_utils

# set dataset name
dataset_name: str = 'kitti_2015'

# set dataset directory
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
_config_parser = utils.load_configuration_file(_config_path)
dataset_dir: str = _config_parser['DATASETS'][dataset_name]
dataset_dict_file: str = _config_parser['DATASET_DICT'][dataset_name]


def _generate_list_with_all_registries(which):
    assert which in ['train', 'test']
    prefix = 'training' if which == 'train' else 'testing'
    if which == 'train':
        li = []
        for i in range(200):
            # info and paths
            index = i
            id = str(i)
            imL = os.path.join(prefix, 'image_2', '{:06d}'.format(i) + '_10.png')
            imR = os.path.join(prefix, 'image_3', '{:06d}'.format(i) + '_10.png')
            gtL = os.path.join(prefix, 'disp_occ_0', '{:06d}'.format(i) +
                               '_10.png')
            gtR = os.path.join(prefix, 'disp_occ_1', '{:06d}'.format(i) +
                               '_10.png')

            # add to list
            dic = {'index': index, 'id': id, 'imL': imL, 'imR': imR,
                   'gtL': gtL, 'gtR': gtR, 'dataset': dataset_name}
            li.append(dic)
        return li
    else:
        li = []
        for i in range(200):
            # info and paths
            index = i
            id = str(i)
            imL = os.path.join(prefix, 'image_2', '{:06d}'.format(i) + '_10.png')
            imR = os.path.join(prefix, 'image_3', '{:06d}'.format(i) + '_10.png')
            gtL = None
            gtR = None

            # add to list
            dic = {'index': index, 'id': id, 'imL': imL, 'imR': imR,
                   'gtL': gtL, 'gtR': gtR, 'dataset': dataset_name}
            li.append(dic)
        return li


class split_dataset(abc.SplitDataset):

    def _load_dict_with_all_registries(self):
        return kitti_utils._load_dict_with_all_registries(dataset_dict_file)

    def _create_list(self, which):
        return _generate_list_with_all_registries(which)

    def load_registry(self, dic):
        return kitti_utils._load_registry(dataset_dir, dataset_name, dic)
