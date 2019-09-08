import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import configparser

# define colorstyle
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = 'gray'


def plot_im(im, title):
    # PIL -> np
    im = np.ascontiguousarray(im)
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    plt.show(block=False)


def plot_disp_map(im, title):
    plt.figure()
    plt.imshow(im)
    plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.show(block=False)


def split(total_train, total_test, tr, val, te):
    assert len(total_train) >= tr + val
    assert len(total_test) >= te
    shuffle(total_train)
    shuffle(total_test)
    train = total_train[:tr] if tr > 0 else []
    val = total_train[tr:tr+val] if val > 0 else []
    test = total_test[:te] if te > 0 else []
    return train, val, test


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return (np.round(gray)).astype(np.uint8)


def conf_file_as_dict(path2file):
    conf = configparser.ConfigParser()
    conf.read(path2file)
    return conf._sections
