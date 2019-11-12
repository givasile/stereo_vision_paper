import os
import pickle
from typing import List, Dict
from PIL import Image
import numpy as np
import raw_dataset.utils as utils
import raw_dataset.abstract_classes as abc
import raw_dataset.kitti_utils as kitti_utils

# set dataset name
dataset_name: str = 'kitti_2012'

# set dataset directory
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
_config_parser = utils.load_configuration_file(_config_path)
dataset_dir: str = _config_parser['DATASETS'][dataset_name]
dataset_dict_file: str = _config_parser['DATASET_DICT'][dataset_name]


def _generate_list_with_all_registries(train_or_test: str) -> List[Dict]:
    assert train_or_test in ['train', 'test']
    prefix = 'training' if train_or_test == 'train' else 'testing'
    nof_examples = 194 if train_or_test == 'train' else 195
    examples_list = []
    for i in range(nof_examples):
        index = i
        id = str(i)
        imL = os.path.join(prefix, 'colored_0', '{:06d}'.format(i) + '_10.png')
        imR = os.path.join(prefix, 'colored_1', '{:06d}'.format(i) + '_10.png')
        dispL = os.path.join(prefix, 'disp_occ', '{:06d}'.format(i) + '_10.png') if train_or_test == 'train' else None
        dispR = None

        example_dict = {'index': index, 'id': id,
                        'imL': imL, 'imR': imR, 'dispL': dispL, 'dispR': dispR,
                        'dataset': dataset_name}
        examples_list.append(example_dict)
    return examples_list


class SplitDataset(abc.SplitDataset):

    def _load_dict_with_all_registries(self):
        return kitti_utils._load_dict_with_all_registries(dataset_dict_file)

    def _create_list(self, which):
        return _generate_list_with_all_registries(which)

    def load_registry(self, dic):
        return kitti_utils._load_registry(dataset_dir, dataset_name, dic)
