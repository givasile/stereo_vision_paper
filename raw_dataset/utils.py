import random
from typing import List, Dict, Tuple
from configparser import ConfigParser
import copy
import matplotlib.pyplot as plt
import numpy as np
import configparser

# define colorstyle
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = 'gray'


def load_configuration_file(path: str) -> ConfigParser:
    # TODO add checks
    config_parser: ConfigParser = configparser.ConfigParser()
    config_parser.read(path)
    return config_parser


def create_relative_paths(dataset_name: str, config_parser: ConfigParser) -> Dict:
    '''
    Create a dictionary with the relative paths to each example of the training and test set.
    Args:
        config_parser: ConfigParser object
        dataset_name: str

    Returns: Dict

    '''

    assert dataset_name in config_parser['DATASETS']

    path_prefix = config_parser['DATASET'][dataset_name]
    if dataset_name == 'kitti_2012':
        pass


def remove_absolute_part_from_paths(registry_dic: dict) -> dict:
    """
    Ugly patch for removing absolute paths from old versioned merged_dataset instances. If merged_dataset already has
    relative paths, everything remains unchanged.
    Args:
        registry_dic: dict describing a registry
    """
    assert 'dataset' in registry_dic.keys()
    if 'freiburg' in registry_dic['dataset'] and '/media/givasile' in registry_dic['imL']:
        for im in ['imL', 'imR', 'dispL', 'dispR']:
            registry_dic[im] = registry_dic[im].split("Freiburg_Synthetic/")[1]
    elif 'kitti_2015' in registry_dic['dataset'] and '/home/givasile' in registry_dic['imL']:
        for im in ['imL', 'imR', 'gtL', 'gtR']:
            if registry_dic[im] is not None:
                registry_dic[im] = registry_dic[im].split("data_scene_flow/")[1]
    elif 'kitti_2012' in registry_dic['dataset'] and '/home/givasile' in registry_dic['imL']:
        for im in ['imL', 'imR', 'gtL', 'gtR']:
            if registry_dic[im] is not None:
                registry_dic[im] = registry_dic[im].split("KITTI_2012/")[1]
    return registry_dic


def plot_im(im, title):
    # PIL -> np
    im = np.ascontiguousarray(im)
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    plt.show(block=False)


def plot_disp_map(im: np.ndarray, title: str):
    plt.figure()
    plt.imshow(im)
    plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.show(block=False)


def split_dataset(training_set: List[Dict], test_set: List[Dict],
                  nof_tr_examples: int, nof_val_examples: int, nof_test_examples: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    # checks
    assert len(training_set) >= nof_tr_examples + nof_val_examples
    assert len(test_set) >= nof_test_examples

    # add randomness on selection
    training_set_1 = copy.deepcopy(training_set)
    test_set_1 = copy.deepcopy(test_set)
    random.shuffle(training_set_1)
    random.shuffle(test_set_1)

    # select examples
    train_examples: List[Dict] = training_set_1[:nof_tr_examples] if nof_tr_examples > 0 else []
    val_examples: List[Dict] = training_set_1[nof_tr_examples:nof_tr_examples + nof_val_examples] if nof_val_examples > 0 else []
    test_examples: List[Dict] = test_set_1[:nof_test_examples] if nof_test_examples > 0 else []
    return train_examples, val_examples, test_examples


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return (np.round(gray)).astype(np.uint8)


def conf_file_as_dict(path2file):
    conf = configparser.ConfigParser()
    conf.read(path2file)
    return conf._sections
