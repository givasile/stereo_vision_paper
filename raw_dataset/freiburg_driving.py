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
dataset_name = 'freiburg_driving'
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

        driving_dir = os.path.join(filepath, 'driving_frames_cleanpass/')
        driving_disp = os.path.join(filepath, 'driving_disparity/')

        subdir1 = ['15mm_focallength', '15mm_focallength']
        subdir2 = ['scene_backwards', 'scene_forwards']
        subdir3 = ['fast', 'slow']

        index = 0
        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
                    for im in imm_l:
                        if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                            imL = driving_dir+i+'/'+j+'/'+k+'/left/'+im
                            id = i+'/'+j+'/'+k+'/left/'+im
                        dispL = driving_disp+i+'/'+j+'/' + \
                            k+'/left/'+im.split(".")[0]+'.pfm'
                        dispR = driving_disp+i+'/'+j+'/' + \
                            k+'/right/'+im.split(".")[0]+'.pfm'
                        if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                            imR = driving_dir+i+'/'+j+'/'+k+'/right/'+im

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
                   dataset='freiburg_driving')
    return tmp


class split_dataset(abs.split_dataset):

    def _create_list(self, which):
        return _create_list(which)

    # funcs
    def load_registry(self, dic):
        return _load_registry(dic)
