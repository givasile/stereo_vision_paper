import os
import numpy as np
import importlib
import pathlib
from typing import List, Dict, Union
import raw_dataset.utils as utils

# load conf file from parent dir
CONF_PATH = pathlib.Path(__file__).resolve().parents[1] / 'conf.ini'
conf = utils.conf_file_as_dict(CONF_PATH.as_posix())


def check_existence(dic: Dict[str, List]) -> None:
    for key, val in dic.items():
        name = key
        message = 'Dataset ' + name + ' does not exist in configuration file!'
        assert name in conf['DATASETS'].keys(), message


class Dataset:
    def __init__(self, dataset_mixture: Union[Dict[str, List], None], create_load: str, state_dict: Union[None, Dict]) -> None:
        """
        dataset_mixture: dictionary expressing how many examples of each dataset will be used ({<dataset_name>:[<train>, <eval>, <test>]} eg. {"kitti_2015": [10, 20, 10], ...} ) 
        """
        assert create_load in ['create', 'load']
        if create_load == 'create':
            # check all asked datasets are defined in the conf file
            check_existence(dataset_mixture)

        # import modules, create split_dataset instances and
        # store as attributes
        self.train: List[Dict] = []
        self.val: List[Dict] = []
        self.test: List[Dict] = []
        self.split: np.ndarray = np.zeros(3)

        if create_load == 'create':
            for key, val in dataset_mixture.items():
                name = key
                split = val

                # dynamically import module
                mod = importlib.import_module('raw_dataset.' + name)

                # initialise instance of raw dataset
                split_dataset = mod.SplitDataset(split)

                # set instance as attribute
                setattr(self, str(name+'_split'), split_dataset)

                # add to sets
                self.train += split_dataset.train
                self.val += split_dataset.val
                self.test += split_dataset.test

                # create split attr
                dat_split = split_dataset.split
                self.split += np.array(dat_split)

            self.split = self.split.astype(np.int).tolist()
        else:
            for key, val in state_dict.items():
                name = key.split('_split')[0]
                # dynamically import module
                mod = importlib.import_module('raw_dataset.' + name)

                # initialise instance of raw dataset by loading
                split_dataset = mod.SplitDataset(None, 'load', val)

                # set instance as attribute
                setattr(self, str(name+'_split'), split_dataset)

                # add to sets
                self.train += split_dataset.train
                self.val += split_dataset.val
                self.test += split_dataset.test

                # create split attr
                dat_split = split_dataset.split
                self.split += np.array(dat_split)

            self.split = self.split.astype(np.int).tolist()

    def load(self, index: int, which_set: str):
        assert which_set in ['train', 'val', 'test']

        # check that index is inside dataset limits
        if which_set == 'train':
            assert self.split[0] > index
        if which_set == 'val':
            assert self.split[1] > index
        if which_set == 'test':
            assert self.split[2] > index

        # get example dictionary
        example_dict = getattr(self, which_set)[index]

        # get the appropriate instance
        name = str(example_dict['dataset']) + '_split'
        dataset_instance = getattr(self, name)
        
        # patch that removes absolute paths from old versioned registries
        example_dict = utils.remove_absolute_part_from_paths(example_dict)

        # add prefixes
        if 'kitti' in example_dict['dataset']:
            for im in ['imL', 'imR', 'gtL', 'gtR']:
                if example_dict[im] is not None:
                    example_dict[im] = os.path.join(conf['DATASETS'][example_dict['dataset']], example_dict[im])
        if 'freiburg' in example_dict['dataset']:
            for im in ['imL', 'imR', 'dispL', 'dispR']:
                if example_dict[im] is not None:
                    example_dict[im] = os.path.join(conf['DATASETS'][example_dict['dataset']], example_dict[im])

        return dataset_instance.load_registry(example_dict)
