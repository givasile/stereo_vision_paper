import os
from typing import List, Dict
import raw_dataset.utils as utils
import raw_dataset.abstract_classes as abc
import raw_dataset.freiburg_utils as freiburg_utils

# set dataset name
dataset_name: str = 'freiburg_monkaa'

# set dataset directory
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
_config_parser = utils.load_configuration_file(_config_path)
dataset_dir: str = _config_parser['DATASETS'][dataset_name]
dataset_dict_file: str = _config_parser['DATASET_DICT'][dataset_name]


def _generate_list_with_all_registries(which: str) -> List[Dict]:
    assert which in ['train', 'test']

    if which == 'train':
        li = []
        # base path for imL, imR
        monkaa_path = os.path.join(dataset_dir, 'monkaa_frames_cleanpass/')
        # base path for dispL, dispR
        monkaa_disp = os.path.join(dataset_dir, 'monkaa_disparity/')
        # all image families
        monkaa_dir = os.listdir(monkaa_path)

        index = 0
        # for all families
        for dd in monkaa_dir:
            # for all images in family
            all_ims = os.listdir(os.path.join(monkaa_path, dd, 'left'))
            all_ims.sort()
            for im in all_ims:
                if freiburg_utils._is_image_file(os.path.join(monkaa_path, dd, 'left', im)):
                    # base path for family
                    im_base = monkaa_path + dd
                    disp_base = monkaa_disp + dd
                    # compute all paths
                    imL = os.path.join(im_base, 'left', im)
                    imR = os.path.join(im_base, 'right', im)
                    dispL = os.path.join(disp_base, 'left', im.split(".")[0] +
                                         '.pfm')
                    dispR = os.path.join(disp_base, 'right', im.split(".")[0] +
                                         '.pfm')
                    id = os.path.join(dd, 'left', im)
                    # create dict
                    dic = {'index': index, 'id': id,
                           'imL': imL.split('Freiburg_Synthetic/')[1],
                           'imR': imR.split('Freiburg_Synthetic/')[1],
                           'dispL': dispL.split('Freiburg_Synthetic/')[1],
                           'dispR': dispR.split('Freiburg_Synthetic/')[1],
                           'dataset': dataset_name}
                    li.append(dic)
                    index += 1
        return li
    else:
        return []


class split_dataset(abc.SplitDataset):

    def _load_dict_with_all_registries(self):
        return freiburg_utils._load_dict_with_all_registries(dataset_dict_file)

    def _create_list(self, which):
        return _generate_list_with_all_registries(which)

    def load_registry(self, dic):
        return freiburg_utils._load_registry(dataset_dir, dataset_name, dic)
