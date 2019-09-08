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
    def __init__(self, dic):
        check_existence(conf, dic)

        # import modules, create split_dataset instances and
        # store as attributes
        self.train = []
        self.val = []
        self.test = []
        self.split = np.zeros(3)
        for i in dic:
            name = list(i.keys())[0]
            split = i[name]
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
        
        # if we are on c0-5 of mug cluster
        if platform.node() == 'compute-0-5.local':
            for split in ['train', 'val', 'test']:
                dataset = getattr(dataset_instance, split)
                if len(dataset) > 0 and dataset[0]['imL'][:6] == "/media":
                    for i, item in enumerate(dataset):
                        dataset[i]['imL'] = dataset[i]['imL'].replace("/media/givasile/givasile/", "/state/partition1/givasile/") 
                        dataset[i]['imR'] = dataset[i]['imR'].replace("/media/givasile/givasile/", "/state/partition1/givasile/") 
                        dataset[i]['dispL'] = dataset[i]['dispL'].replace("/media/givasile/givasile/", "/state/partition1/givasile/")
                        dataset[i]['dispR'] = dataset[i]['dispR'].replace("/media/givasile/givasile/", "/state/partition1/givasile/") 
        # if we are on athos
        elif platform.node() == 'athos':
            for split in ['train', 'val', 'test']:
                dataset = getattr(dataset_instance, split)
                if len(dataset) > 0 and dataset[0]['imL'][:6] == "/media":
                    for i, item in enumerate(dataset):
                        dataset[i]['imL'] = dataset[i]['imL'].replace("/media/givasile/givasile/", "/home/givasile/stereo_vision/data/givasile/") 
                        dataset[i]['imR'] = dataset[i]['imR'].replace("/media/givasile/givasile/", "/home/givasile/stereo_vision/data/givasile/") 
                        dataset[i]['dispL'] = dataset[i]['dispL'].replace("/media/givasile/givasile/", "/home/givasile/stereo_vision/data/givasile/")
                        dataset[i]['dispR'] = dataset[i]['dispR'].replace("/media/givasile/givasile/", "/home/givasile/stereo_vision/data/givasile/") 
        else:
            for split in ['train', 'val', 'test']:
                dataset = getattr(dataset_instance, split)
                if len(dataset) > 0 and dataset[0]['imL'][:6] == "/media":
                    for i, item in enumerate(dataset):
                        dataset[i]['imL'] = dataset[i]['imL'].replace("/media/givasile/givasile/datasets/stereo_vision/raw/Freiburg_Synthetic/", conf['DATASETS']['freiburg'])
                        dataset[i]['imR'] = dataset[i]['imR'].replace("/media/givasile/givasile/datasets/stereo_vision/raw/Freiburg_Synthetic/", conf['DATASETS']['freiburg'])
                        dataset[i]['dispL'] = dataset[i]['dispL'].replace("/media/givasile/givasile/datasets/stereo_vision/raw/Freiburg_Synthetic/", conf['DATASETS']['freiburg'])
                        dataset[i]['dispR'] = dataset[i]['dispR'].replace("/media/givasile/givasile/datasets/stereo_vision/raw/Freiburg_Synthetic/", conf['DATASETS']['freiburg'])
                        
            
        return dataset_instance.load_registry(dic)
