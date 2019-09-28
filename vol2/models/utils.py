import raw_dataset.merged_dataset as merged_dataset
import os
import pickle
import importlib
import torch


def load_common_dataset(dataset_name: str, common_dataset_dir: str) -> merged_dataset.Dataset:
    with open(os.path.join(common_dataset_dir, dataset_name + '.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)


def load_specific_dataset(dataset_dir: str) -> merged_dataset.Dataset:
    with open(os.path.join(dataset_dir, 'merged_dataset_state_dict.pickle'), 'rb') as fm:
        dataset_state_dict = pickle.load(fm)
    return merged_dataset.Dataset(None, 'load', dataset_state_dict)


def load_modules(cnn_name: str):
    submodules = importlib.import_module('vol2.models.' + cnn_name + '.submodules')
    net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
    merged = importlib.import_module('raw_dataset.merged_dataset')
    preprocess = importlib.import_module('preprocess')
    evaluate = importlib.import_module('evaluate')
    visualize = importlib.import_module('visualize')
    return submodules, net, merged, preprocess, evaluate, visualize


def load_things_for_inspection(cnn_name, conf_file, checkpoint_n, experiment_n, get_common_dataset, common_dataset_name, device, which, form, limit_maxD, example_num):
    # import all needed modules
    submodules, net, merged, preprocess, evaluate, visualize = load_modules(cnn_name)

    # create helpful paths
    experiment_directory = os.path.join(conf_file['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n))
    checkpoint_filepath = os.path.join(experiment_directory, 'checkpoint_' + str(checkpoint_n) + '.tar')

    # load dataset
    if get_common_dataset:
        merged_dataset = load_common_dataset(common_dataset_name, conf_file['PATHS']['common_datasets'])
    else:
        merged_dataset = load_specific_dataset(experiment_directory)

    # create model
    if device == 'cpu':
        model_instance = net.model()
    else:
        model_instance = net.model().cuda()

    # restore weights
    checkpoint = torch.load(checkpoint_filepath)
    model_instance.load_state_dict(checkpoint['state_dict'])

    # restore training statistics
    stats = checkpoint['stats']

    data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
    imL, imR, dispL, maskL = data_feeder[example_num]
    imL = imL.unsqueeze(0).cuda()
    imR = imR.unsqueeze(0).cuda()
    max_limit = dispL.max()
    dispL = dispL.unsqueeze(0).cuda()
    maskL = maskL.type(torch.bool).unsqueeze(0).cuda()

    return imL, imR, dispL, maskL, model_instance




