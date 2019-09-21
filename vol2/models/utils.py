import raw_dataset.merged_dataset as merged_dataset
import os
import pickle


def load_common_dataset(dataset_name: str, common_dataset_dir: str) -> merged_dataset.Dataset:
    with open(os.path.join(common_dataset_dir, dataset_name + '.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)


def load_specific_dataset(dataset_dir: str) -> merged_dataset.Dataset:
    with open(os.path.join(dataset_dir, 'merged_dataset_state_dict.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)

