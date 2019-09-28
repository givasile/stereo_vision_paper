import sys
import configparser
import importlib
import torch
import time
import os

# configuration variables
cnn_name: str = 'simple_net'

experiment_n = 1
checkpoint_n = 10
which = 'test'; assert which in ['train', 'val', 'test']
form = 'full_im'; assert form in ['crop', 'full_im']
limit_maxD = True  # extract statistics only on disp_gt < maxD
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 100
device = 'cuda'
maxD = 192
mode = 'train'  # ['train', 'eval']


def run(conf_file):
    # import all needed modules
    utils = importlib.import_module('vol2.models.utils')
    net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
    preprocess = importlib.import_module('preprocess')
    evaluate = importlib.import_module('evaluate')
    visualize = importlib.import_module('visualize')

    # create helpful paths
    experiment_directory = os.path.join(conf_file['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n))
    checkpoint_filepath = os.path.join(experiment_directory, 'checkpoint_' + str(checkpoint_n) + '.tar')

    # load dataset
    if get_common_dataset:
        merged_dataset = utils.load_common_dataset(common_dataset_name, conf_file['PATHS']['common_datasets'])
    else:
        merged_dataset = utils.load_specific_dataset(experiment_directory)

    # create model
    if device == 'cpu':
        model_instance = net.model()
    else:
        model_instance = net.model().cuda()

    # restore weights
    checkpoint = torch.load(checkpoint_filepath)
    model_instance.load_state_dict(checkpoint['state_dict'])

    # prepare image
    data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
    imL, imR, dispL, maskL = data_feeder[example_num]
    imL = imL.unsqueeze(0).cuda()
    imR = imR.unsqueeze(0).cuda()
    dispL = dispL.unsqueeze(0).cuda()
    maskL = maskL.type(torch.bool).unsqueeze(0).cuda()

    print('Cnn name: %s, number of parameters: %d' %(cnn_name,
        sum([p.data.nelement() for p in model_instance.parameters()])))

    # forward pass
    start_time = time.time()
    tmp = net.inspection(model_instance, device, mode, imL, imR, dispL, maskL, maxD)
    if tmp is not None:
        pred, imL_d, imR_d, cost_a, cost_b, mae, std_ae, pcg = tmp

    end_time = time.time()
    elapsed_time = end_time - start_time

    # error
    threshold = 3
    err_im = evaluate.image_absolute_error(pred, dispL, maskL)
    over_thres_im = evaluate.image_percentage_over_limit(pred, dispL, maskL, threshold)
    print(' Time: %.3f \n mean error: %.3f px \n percentage over %d pixel: %f'
          % (elapsed_time, mae, threshold, pcg))

    # visualize
    visualize.imshow_mch(imL[0].cpu(), 0, 'imL')
    visualize.imshow_mch(imR[0].cpu(), 0, 'imR')
    visualize.imshow_1ch(pred[0].cpu(), 'prediction', [
                         dispL.min().item(), dispL.max().item()])
    visualize.imshow_1ch(dispL[0].cpu(), 'ground truth')
    visualize.imshow_1ch(over_thres_im[0].cpu(), 'error over threshold')
    visualize.imshow_1ch(err_im[0].cpu(), 'abs error', (0,30))


if __name__ == '__main__':
    # import config file
    conf_path = './../../conf.ini'
    conf = configparser.ConfigParser()
    conf.read(conf_path)

    # add parent path, if not already added
    parent_path = conf['PATHS']['parent_dir']
    sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

    # run main
    run(conf)