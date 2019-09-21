import os
from PIL import Image
import pickle
import numpy as np
import raw_dataset.abstract_classes as abc


def _load_registry(dataset_dir, dataset_name, example_dict):
    class Registry(abc.Registry):
        pass

    def gt2mask(x):
        y = np.zeros_like(x)
        y[x > 0.0] = 1
        return y.astype(np.uint8)

    # load
    imL_rgb = Image.open(os.path.join(dataset_dir, example_dict['imL'])).convert('RGB')
    imR_rgb = Image.open(os.path.join(dataset_dir, example_dict['imR'])).convert('RGB')

    # rgb -> gray
    imL_g = imL_rgb.convert('LA')
    imR_g = imR_rgb.convert('LA')

    # load ground truth
    word = 'gtL' if 'gtL' in example_dict.keys() else 'dispL'
    if example_dict[word] is not None:
        gtL = np.array(Image.open(os.path.join(dataset_dir, example_dict[word])), dtype=np.float32) / 256.0
        mask1 = gt2mask(gtL)
        maxDL = np.max(gtL)
    else:
        gtL = None
        mask1 = None
        maxDL = None

    word = 'gtR' if 'gtR' in example_dict.keys() else 'dispR'
    if example_dict['gtR'] is not None:
        gtR = np.array(Image.open(os.path.join(dataset_dir, example_dict[word])), dtype=np.float32) / 256.0
        mask2 = gt2mask(gtR)
        maxDR = np.max(gtR)
    else:
        gtR = None
        mask2 = None
        maxDR = None

    # return registry instance
    tmp = Registry(example_dict['index'], example_dict['id'], imL_g, imR_g, imL_rgb,
                   imR_rgb, gtL, gtR, mask1, mask2, maxDL, maxDR,
                   dataset=dataset_name)
    return tmp


def _load_dict_with_all_registries(dataset_dict_file):
    if dataset_dict_file != 'None':
        with open(dataset_dict_file, 'rb') as fm:
            return pickle.load(fm)
    else:
        return None
