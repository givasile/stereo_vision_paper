import os
from typing import List, Dict
import raw_dataset.utils as utils
import raw_dataset.abstract_classes as abc
import raw_dataset.freiburg_utils as freiburg_utils

# set dataset name
dataset_name: str = 'freiburg_flying'

# set dataset directory
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
_config_parser = utils.load_configuration_file(_config_path)
dataset_dir: str = _config_parser['DATASETS'][dataset_name]
dataset_dict_file: str = _config_parser['DATASET_DICT'][dataset_name]


def _generate_list_with_all_registries(which: str) -> List[Dict]:
    assert which in ['train', 'test']
    # base path for imL, imR
    flying_path = os.path.join(dataset_dir, 'flying_frames_cleanpass')
    # base path for dispL, dispR
    flying_disp = os.path.join(dataset_dir, 'flying_disparity/')
    # train directory
    if which == 'train':
        li = []
        flying_dir = os.path.join(flying_path, 'TRAIN')
        flying_dir1 = os.path.join(flying_disp, 'TRAIN')
        # subdir
        subdir = ['A', 'B', 'C']
        index = 0
        # for all basic subdirectories
        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))
            flying.sort()
            for ff in flying:
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff,  'left'))
                imm_l.sort()
                for im in imm_l:
                    flying_dir_a = flying_dir.split('/Freiburg_Synthetic/')[1]
                    flying_dir1_a = flying_dir1.split('/Freiburg_Synthetic/')[1]
                    if freiburg_utils._is_image_file(os.path.join(flying_dir, ss, ff, 'left', im)):
                        imL = os.path.join(flying_dir_a, ss, ff, 'left', im)
                        imR = os.path.join(flying_dir_a, ss, ff, 'right', im)
                        dispL = os.path.join(flying_dir1_a, ss, ff, 'left',
                                             im.split(".")[0]+'.pfm')
                        dispR = os.path.join(flying_dir1_a, ss, ff, 'right',
                                             im.split(".")[0]+'.pfm')
                        identity = os.path.join('TRAIN', ss, ff, 'left', im)
                        # create dict
                        dic = {'index': index, 'id': identity, 'imL': imL,
                               'imR': imR, 'dispL': dispL, 'dispR': dispR,
                               'dataset': dataset_name}
                        li.append(dic)
                        index += 1
    else:
        li = []
        flying_dir = os.path.join(flying_path, 'TEST')
        flying_dir1 = os.path.join(flying_disp, 'TEST')
        # subdir
        subdir = ['A', 'B', 'C']
        index = 0
        # for all basic subdirectories
        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))
            flying.sort()
            for ff in flying:
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff,  'left'))
                imm_l.sort()
                for im in imm_l:
                    term1 = freiburg_utils._is_image_file(os.path.join(flying_dir, ss, ff,
                                                       'left', im))
                    term2 = freiburg_utils._is_image_file(os.path.join(flying_dir, ss, ff,
                                                       'right', im))
                    if term1 and term2:
                        flying_dir_a = flying_dir.split('/Freiburg_Synthetic/')[1]
                        flying_dir1_a = flying_dir1.split('/Freiburg_Synthetic/')[1]
                        imL = os.path.join(flying_dir_a, ss, ff, 'left', im)
                        imR = os.path.join(flying_dir_a, ss, ff, 'right', im)
                        dispL = os.path.join(flying_dir1_a, ss, ff, 'left',
                                             im.split(".")[0]+'.pfm')
                        dispR = os.path.join(flying_dir1_a, ss, ff, 'right',
                                             im.split(".")[0]+'.pfm')
                        identity = os.path.join('TEST', ss, ff, 'left', im)
                        # create dict
                        dic = {'index': index, 'id': identity, 'imL': imL,
                               'imR': imR, 'dispL': dispL, 'dispR': dispR,
                               'dataset': dataset_name}
                        li.append(dic)
                        index += 1
    return li


class split_dataset(abc.SplitDataset):

    def _load_dict_with_all_registries(self):
        return freiburg_utils._load_dict_with_all_registries(dataset_dict_file)

    def _create_list(self, which):
        return _generate_list_with_all_registries(which)

    def load_registry(self, dic):
        return freiburg_utils._load_registry(dataset_dir, dataset_name, dic)
