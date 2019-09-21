import os
import raw_dataset.utils as utils
import raw_dataset.abstract_classes as abc
import raw_dataset.freiburg_utils as freiburg_utils

# set dataset name
dataset_name: str = 'freiburg_driving'

# set dataset directory
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
_config_parser = utils.load_configuration_file(_config_path)
dataset_dir: str = _config_parser['DATASETS'][dataset_name]
dataset_dict_file: str = _config_parser['DATASET_DICT'][dataset_name]


def _generate_list_with_all_registries(which):
    assert which in ['train', 'test']

    if which == 'train':
        li = []

        driving_dir = os.path.join(dataset_dir, 'driving_frames_cleanpass/')
        driving_disp = os.path.join(dataset_dir, 'driving_disparity/')

        subdir1 = ['15mm_focallength', '15mm_focallength']
        subdir2 = ['scene_backwards', 'scene_forwards']
        subdir3 = ['fast', 'slow']

        index = 0
        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
                    for im in imm_l:
                        if freiburg_utils._is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                            imL = driving_dir+i+'/'+j+'/'+k+'/left/'+im
                            id = i+'/'+j+'/'+k+'/left/'+im
                        dispL = driving_disp+i+'/'+j+'/' + \
                            k+'/left/'+im.split(".")[0]+'.pfm'
                        dispR = driving_disp+i+'/'+j+'/' + \
                            k+'/right/'+im.split(".")[0]+'.pfm'
                        if freiburg_utils._is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                            imR = driving_dir+i+'/'+j+'/'+k+'/right/'+im

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
