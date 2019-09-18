import numpy as np
import importlib
import pathlib
import platform
import raw_dataset.utils as utils

# load conf file from parent dir
CONF_PATH = pathlib.Path(__file__).resolve().parents[1] / 'conf.ini'
conf = utils.conf_file_as_dict(CONF_PATH.as_posix())


def check_existence(conf, dic):
    for i in dic:
        name = list(i.keys())[0]
        message = 'Dataset ' + name + ' does not exist!'
        assert name in conf['DATASETS'].keys(), message


class dataset:
    def __init__(self, dataset_mixture):
        """
        dataset_mixture: dictionary expressing how many examples of each dataset will be used ({<dataset_name>:[<train>, <eval>, <test>]} eg. {"kitti_2015": [10, 20, 10], ...} ) 
        """

        # check all asked datasets are defined in the conf file
        check_existence(conf, dataset_mixture)

        # import modules, create split_dataset instances and
        # store as attributes
        self.train = []
        self.val = []
        self.test = []
        self.split = np.zeros(3)
        for key, val in dataset_mixture:
            name = key
            split = val
            mod = importlib.import_module('raw_dataset.' + name)

            # set instances as attributes
            split_instance = mod.split_dataset(split)
            setattr(self, str(name+'_split'), split_instance)

            # add to sets
            self.train += getattr(split_instance, 'train')
            self.val += getattr(split_instance, 'val')
            self.test += getattr(split_instance, 'test')

            # create split attr
            dat_split = getattr(split_instance, 'split')
            self.split += np.array(dat_split)

        self.split = self.split.astype(np.int).tolist()

    def load(self, reg, which):
        assert which in ['train', 'val', 'test']

        # check and find appropriate dict and name of load func
        if which == 'train':
            assert self.split[0] > reg
        if which == 'val':
            assert self.split[1] > reg
        if which == 'test':
            assert self.split[2] > reg

        dic = getattr(self, which)[reg]
        name = str(dic['dataset']) + '_split'
        dataset_instance = getattr(self, name)
        
        # Patch for on-the-fly fixing of wrong paths
        if platform.node() == 'compute-0-5.local':
            if 'freiburg' in dic['dataset']:
                pref_new = "/state/partition1/givasile/"
                pref_old = dic['imR'].split("Freiburg_Synthetic")[0] + "Freiburg_Synthetic/"
                dic['imL'] = dic['imL'].replace(pref_old, pref_new)
                dic['imR'] = dic['imR'].replace(pref_old, pref_new)
                dic['dispL'] = dic['dispL'].replace(pref_old, pref_new)
                dic['dispR'] = dic['dispR'].replace(pref_old, pref_new)
        else:
            if 'freiburg' in dic['dataset']:
                pref_new = conf['DATASETS']['freiburg']
                pref_old = dic['imR'].split("Freiburg_Synthetic")[0] + "Freiburg_Synthetic/"
                dic['imL'] = dic['imL'].replace(pref_old, pref_new)
                dic['imR'] = dic['imR'].replace(pref_old, pref_new)
                dic['dispL'] = dic['dispL'].replace(pref_old, pref_new)
                dic['dispR'] = dic['dispR'].replace(pref_old, pref_new)

        return dataset_instance.load_registry(dic)
