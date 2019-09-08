import os
import sys
import importlib
import configparser
from PIL import Image
import numpy as np


# load conf file from parent dir
CONF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'conf.ini')
conf = configparser.ConfigParser()
conf.read(CONF_PATH)
conf = conf._sections

# add parent path of project
parent = conf['PATHS']['parent_dir']
ins = sys.path.insert(1, parent)
ins if parent not in sys.path else 0

# import custom modules
abs = importlib.import_module('raw_dataset.abstract_classes')

# parent path
dataset_name = 'kitti_2015'
PARENT_DIR = conf['DATASETS'][dataset_name]


def gt2mask(x):
    y = np.zeros_like(x)
    y[x > 0.0] = 1
    return y.astype(np.uint8)


class registry(abs.registry):
    pass


def _create_list(which):
    assert which in ['train', 'test']
    if which == 'train':
        name1 = 'training'
        tmp = os.path.join(PARENT_DIR, 'data_scene_flow', name1)
        li = []
        for i in range(200):
            # info and paths
            index = i
            id = i
            imL = os.path.join(tmp, 'image_2', '{:06d}'.format(i) + '_10.png')
            imR = os.path.join(tmp, 'image_3', '{:06d}'.format(i) + '_10.png')
            gtL = os.path.join(tmp, 'disp_occ_0', '{:06d}'.format(i) +
                               '_10.png')
            gtR = os.path.join(tmp, 'disp_occ_1', '{:06d}'.format(i) +
                               '_10.png')

            # add to list
            dic = {'index': index, 'id': id, 'imL': imL, 'imR': imR,
                   'gtL': gtL, 'gtR': gtR, 'dataset': dataset_name}
            li.append(dic)
        return li
    else:
        name1 = 'testing'
        tmp = os.path.join(PARENT_DIR, 'data_scene_flow', name1)
        li = []
        for i in range(200):
            # info and paths
            index = i
            id = i
            imL = os.path.join(tmp, 'image_2', '{:06d}'.format(i) + '_10.png')
            imR = os.path.join(tmp, 'image_3', '{:06d}'.format(i) + '_10.png')
            gtL = None
            gtR = None

            # add to list
            dic = {'index': index, 'id': id, 'imL': imL, 'imR': imR,
                   'gtL': gtL, 'gtR': gtR, 'dataset': dataset_name}
            li.append(dic)
        return li


def _load_registry(dic):
    # load
    imL_rgb = Image.open(dic['imL']).convert('RGB')
    imR_rgb = Image.open(dic['imR']).convert('RGB')

    # rgb -> gray
    imL_g = imL_rgb.convert('LA')
    imR_g = imR_rgb.convert('LA')

    # load ground truth
    if dic['gtL'] is not None:
        gtL = np.array(Image.open(dic['gtL']), dtype=np.float32)/256.0
        mask1 = gt2mask(gtL)
        maxDL = np.max(gtL)
    else:
        gtL = None
        mask1 = None
        maxDL = None
    if dic['gtR'] is not None:
        gtR = np.array(Image.open(dic['gtR']), dtype=np.float32)/256.0
        mask2 = gt2mask(gtR)
        maxDR = np.max(gtR)
    else:
        gtR = None
        mask2 = None
        maxDR = None

    # return registry instance
    tmp = registry(dic['index'], dic['id'], imL_g, imR_g, imL_rgb,
                   imR_rgb, gtL, gtR, mask1, mask2, maxDL, maxDR,
                   dataset='kitti_2015')
    return tmp


class split_dataset(abs.split_dataset):

    def _create_list(self, which):
        return _create_list(which)

    # funcs
    def load_registry(self, dic):
        return _load_registry(dic)
