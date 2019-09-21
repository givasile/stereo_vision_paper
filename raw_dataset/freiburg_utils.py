import os
from PIL import Image
import raw_dataset.pfm as pfm
import raw_dataset.abstract_classes as abc
import pickle
import numpy as np


def _load_registry(dataset_dir, dataset_name, dic):
    def _gt2mask(x):
        return np.ones_like(x)

    class Registry(abc.Registry):
        pass

    # load
    imL_rgb = Image.open(os.path.join(dataset_dir, dic['imL'])).convert('RGB')
    imR_rgb = Image.open(os.path.join(dataset_dir, dic['imR'])).convert('RGB')

    # rgb -> gray
    imL_g = imL_rgb.convert('LA')
    imR_g = imR_rgb.convert('LA')

    # load ground truth
    if dic['dispL'] is not None:
        gtL = pfm.readPFM(os.path.join(dataset_dir, dic['dispL']))[0]
        mask1 = _gt2mask(gtL)
        maxDL = gtL.max()
    else:
        gtL = None
        mask1 = None
        maxDL = None
    if dic['dispR'] is not None:
        gtR = pfm.readPFM(os.path.join(dataset_dir, dic['dispR']))[0]
        mask2 = _gt2mask(gtR)
        maxDR = gtR.max()
    else:
        gtR = None
        mask2 = None
        maxDR = None

    # return registry instance
    tmp = Registry(dic['index'], dic['id'], imL_g, imR_g, imL_rgb,
                   imR_rgb, gtL, gtR, mask1, mask2, maxDL, maxDR,
                   dataset=dataset_name)
    return tmp


def _load_dict_with_all_registries(dataset_dict_file):
    if dataset_dict_file != 'None':
        with open(dataset_dict_file, 'rb') as fm:
            return pickle.load(fm)
    else:
        return None



def _is_image_file(filename):
    _img_extensions = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in _img_extensions)
