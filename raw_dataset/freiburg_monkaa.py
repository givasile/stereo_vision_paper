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
pfm = importlib.import_module('raw_dataset.pfm')

# parent path of dataset
dataset_name = 'freiburg_monkaa'
PARENT_DIR = conf['DATASETS'][dataset_name]
filepath = PARENT_DIR

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def gt2mask(x):
    return np.ones_like(x)


def _create_list(which):
    assert which in ['train', 'test']

    if which == 'train':
        li = []
        # base path for imL, imR
        monkaa_path = os.path.join(filepath, 'monkaa_frames_cleanpass/')
        # base path for dispL, dispR
        monkaa_disp = os.path.join(filepath, 'monkaa_disparity/')
        # all image families
        monkaa_dir = os.listdir(monkaa_path)

        index = 0
        # for all families
        for dd in monkaa_dir:
            # for all images in family
            all_ims = os.listdir(os.path.join(monkaa_path, dd, 'left'))
            all_ims.sort()
            for im in all_ims:
                if is_image_file(os.path.join(monkaa_path, dd, 'left', im)):
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
                    dic = {'index': index, 'id': id, 'imL': imL,
                           'imR': imR, 'dispL': dispL, 'dispR': dispR,
                           'dataset': dataset_name}
                    li.append(dic)
                    index += 1
        return li
    else:
        return []


class registry(abs.registry):
    pass


def _load_registry(dic):
    # load
    imL_rgb = Image.open(dic['imL']).convert('RGB')
    imR_rgb = Image.open(dic['imR']).convert('RGB')

    # rgb -> gray
    imL_g = imL_rgb.convert('LA')
    imR_g = imR_rgb.convert('LA')

    # load ground truth
    if dic['dispL'] is not None:
        gtL = pfm.readPFM(dic['dispL'])[0]
        mask1 = gt2mask(gtL)
        maxDL = gtL.max()
    else:
        gtL = None
        mask1 = None
        maxDL = None
    if dic['dispR'] is not None:
        gtR = pfm.readPFM(dic['dispR'])[0]
        mask2 = gt2mask(gtR)
        maxDR = gtR.max()
    else:
        gtR = None
        mask2 = None
        maxDR = None

    # return registry instance
    tmp = registry(dic['index'], dic['id'], imL_g, imR_g, imL_rgb,
                   imR_rgb, gtL, gtR, mask1, mask2, maxDL, maxDR,
                   dataset='freiburg_monkaa')
    return tmp


class split_dataset(abs.split_dataset):

    def _create_list(self, which):
        return _create_list(which)

    # funcs
    def load_registry(self, dic):
        return _load_registry(dic)
