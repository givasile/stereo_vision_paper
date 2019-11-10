import sys
import torch
import configparser
import importlib
import timeit
import random
import os
import math
import time
import numpy as np

cnn_name = 'merging_info_net_custom_features'

# import config file
conf_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
conf_path += '/conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

# import custom modules
submodules = importlib.import_module('vol2.models.' + cnn_name + '.submodules1')
evaluate = importlib.import_module('evaluate')
preprocess = importlib.import_module('preprocess')

# static hyperparameters of cnn
ch1 = 32
num_of_residuals_on_f2 = 4      # doesn't exist
num_of_residuals_on_f3 = 4
num_of_residuals_on_f5 = 2
num_of_residuals_on_f7 = 3
num_of_residuals_on_f8 = 3


# define model class
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features_extraction_2d = submodules.descriptors_extraction(ch1,
                                                                        num_of_residuals_on_f2,
                                                                        num_of_residuals_on_f3)

        self.for_out_3d = submodules.for_out_3d(ch1, num_of_residuals_on_f5)

        self.module_3d = submodules.module_3d(
            ch1, num_of_residuals_on_f7, num_of_residuals_on_f8)

    def forward(self, imL, imR, scales, prediction_from_scales, initial_scale, device, inspection):

        # assert initial_scale given matches input dimensions
        assert imL.shape[2] == initial_scale[1]
        assert imR.shape[3] == initial_scale[2]

        # features extraction
        imL_d = self.features_extraction_2d(
            imL, scales, prediction_from_scales.keys(), device)
        imR_d = self.features_extraction_2d(
            imR, scales, prediction_from_scales.keys(), device)

        # create comparison volumes
        volumes = []
        for i in range(len(imL_d)):
            if imL_d[i] is not None:
                volumes.append(submodules.patch_comparison_volume(
                    imL, imR, imL_d[i], imR_d[i], scales[i][0], self.training, device))
            else:
                volumes.append(None)

        # min key that has after label, if there is not after label assign a big number
        tmp = []
        for key in prediction_from_scales.keys():
            if 'after' in prediction_from_scales[key]:
                tmp.append(key)
        if not tmp:
            min_sc = 10000
        else:
            min_sc = min(tmp)

        # volumes_dict module
        volumes_dict = {}

        # first volume is by default the last volume
        volumes_dict[len(scales) - 1] = {}
        volumes_dict[len(scales) - 1]['before'] = volumes[-1]
        volumes_dict[len(scales) - 1]['after'] = self.module_3d(0, volumes[-1])
        for i in range(len(scales) - 2, -1, -1):
            if i >= min_sc:
                volumes_dict[i] = {}
                volumes_dict[i]['before'] = volumes[i]
                volumes_dict[i]['after'] = self.module_3d(
                    volumes_dict[i+1]['after'], volumes_dict[i]['before'])

            elif i >= min(prediction_from_scales.keys()):
                volumes_dict[i] = {}
                volumes_dict[i]['before'] = volumes[i]

        # for out
        for_out_dict = {}
        for i in prediction_from_scales.keys():
            for_out_dict[i] = {}
            for bef_af in prediction_from_scales[i]:
                for_out_dict[i][bef_af] = self.for_out_3d(
                    volumes_dict[i][bef_af])

        # predictions
        predictions_dict = {}
        for key in for_out_dict.keys():
            predictions_dict[key] = {}
            for bef_af in for_out_dict[key].keys():
                tmp = for_out_dict[key][bef_af]
                predictions_dict[key][bef_af] = submodules.softargmax(
                    tmp, initial_scale, device)

        if inspection:
            return imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict
        else:
            return predictions_dict


def training_step(model_instance, initial_scale, scales, prediction_from_scales,
                  device, optimizer, loss_from_scales_in_training,
                  imL, imR, dispL, maskL):

    if maskL.sum() > 0:

        # forward, backward and update
        model_instance.train()
        optimizer.zero_grad()
        predictions_dict = model_instance(imL, imR, scales, prediction_from_scales,
                                          initial_scale, device, inspection=False)
        loss = 0
        for key in loss_from_scales_in_training.keys():
            for key1 in loss_from_scales_in_training[key].keys():
                tmp = loss_from_scales_in_training[key][key1]
                pred = predictions_dict[key][key1][maskL]
                loss += tmp * \
                    torch.nn.functional.smooth_l1_loss(pred, dispL[maskL])

        loss.backward()
        optimizer.step()

        # statistics
        for key in loss_from_scales_in_training.keys():
            for key1 in loss_from_scales_in_training[key].keys():
                predictions_dict[key][key1] = predictions_dict[key][key1].detach()

        mae = {}
        err_pcg = {}
        threshold = 3
        for key in loss_from_scales_in_training.keys():
            mae[key] = {}
            err_pcg[key] = {}
            for key1 in loss_from_scales_in_training[key].keys():
                mae[key][key1] = evaluate.mean_absolute_error(
                    predictions_dict[key][key1], dispL, maskL)
                err_pcg[key][key1] = evaluate.percentage_over_limit(
                    predictions_dict[key][key1], dispL, maskL, threshold)
        return mae, err_pcg, loss
    else:
        return None, None, None


def training_epoch(merged, batch_size, stats, model_instance, optimizer, initial_scale,
                   scales, prediction_from_scales, loss_from_scales_in_training, device):
    # prepare dataset
    data_feeder = preprocess.dataset(merged, 'train', 'crop', True)
    dataloader = torch.utils.data.DataLoader(data_feeder, batch_size=batch_size,
                                             shuffle=True, num_workers=batch_size)
    epoch = stats['general']['cur_epoch']
    print('Starting epoch %d of training' % (epoch))
    epoch_start_time = time.time()

    # add new epoch on statistics
    stats['train']['loss'].append([])
    for key in loss_from_scales_in_training.keys():
        for key1 in loss_from_scales_in_training[key].keys():
            stats['train']['mae'][str(key)+'_'+key1].append([])
            stats['train']['pcg'][str(key)+'_'+key1].append([])

    # for every batch
    for batch_idx, (imL, imR, dispL, maskL) in enumerate(dataloader):
        if maskL.sum() > 0:
            batch_start_time = time.time()
            imL = imL.cuda()
            imR = imR.cuda()
            dispL = dispL.cuda()
            maskL = maskL.cuda()

            max_disp = 192
            h = imL.shape[2]
            w = imL.shape[3]

            mae, err_pcg, loss = training_step(model_instance, initial_scale, scales,
                                               prediction_from_scales, device, optimizer,
                                               loss_from_scales_in_training,
                                               imL, imR, dispL, maskL)

            # print results
            print('\n')
            print('Iter %d: LOSS = %.3f, time = %.2f' %
                  (batch_idx+1, loss, time.time() - batch_start_time))
            for key in loss_from_scales_in_training.keys():
                for key1 in loss_from_scales_in_training[key].keys():
                    print('training, iter: %d, %s_%s  MAE = %.3f , PCG = %2.f , time = %.2f' %
                          (batch_idx+1, key, key1, mae[key][key1],
                           err_pcg[key][key1], time.time() - batch_start_time))

            # update training statistics
            stats['train']['loss'][epoch-1].append(loss.detach().item())
            for key in loss_from_scales_in_training.keys():
                for key1 in loss_from_scales_in_training[key].keys():
                    stats['train']['mae'][str(
                        key)+'_'+key1][epoch-1].append(mae[key][key1])
                    stats['train']['pcg'][str(
                        key)+'_'+key1][epoch-1].append(err_pcg[key][key1])
            stats['general']['cur_step'] += 1

    # print results after training epoch
    print('\n#### Results on training epoch %d #### \n' % (epoch))
    print('time = %.3f' % (time.time() - epoch_start_time))
    for key in loss_from_scales_in_training.keys():
        for key1 in loss_from_scales_in_training[key].keys():
            print('%s_%s, μ(MAE) = %.3f' %
                  (key, key1, np.mean(stats['train']['mae'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, μ(PCG) = %.2f' %
                  (key, key1, np.mean(stats['train']['pcg'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, σ(MAE) = %.3f' %
                  (key, key1, np.std(stats['train']['mae'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, σ(PCG) = %.2f' %
                  (key, key1, np.std(stats['train']['pcg'][str(key)+'_'+key1][epoch-1])))
    print('\n')


def inspection(model_instance, initial_scale, scales, prediction_from_scales,
               device, imL, imR, dispL, maskL):
    model_instance.eval()

    with torch.no_grad():
        imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = model_instance(
            imL, imR, scales, prediction_from_scales, initial_scale, device, inspection=True)

    for key in prediction_from_scales.keys():
        for key1 in prediction_from_scales[key]:
            predictions_dict[key][key1] = predictions_dict[key][key1].detach()

    if maskL.sum() > 0:
        mae = {}
        err_pcg = {}
        threshold = 3
        print(prediction_from_scales)
        for key in prediction_from_scales.keys():
            mae[key] = {}
            err_pcg[key] = {}
            for key1 in prediction_from_scales[key]:
                mae[key][key1] = evaluate.mean_absolute_error(
                    predictions_dict[key][key1], dispL, maskL)
                err_pcg[key][key1] = evaluate.percentage_over_limit(
                    predictions_dict[key][key1], dispL, maskL, threshold)
                print('mae on %d_%s: %.3f px' % (key, key1, mae[key][key1]))
                print('pcg on %d_%s: %.2f percent' %
                      (key, key1, err_pcg[key][key1]))
    else:
        mae, err_pcg = None, None

    return mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict


def inference(model_instance, initial_scale, scales, prediction_from_scales,
              device, imL, imR, dispL, maskL):
    model_instance.eval()

    with torch.no_grad():
        predictions_dict = model_instance(imL, imR, scales, prediction_from_scales,
                                          initial_scale, device, inspection=False)

    for key in prediction_from_scales.keys():
        for key1 in prediction_from_scales[key]:
            predictions_dict[key][key1] = predictions_dict[key][key1].detach()

    if maskL.sum() > 0:
        mae = {}
        err_pcg = {}
        threshold = 3
        for key in prediction_from_scales.keys():
            mae[key] = {}
            err_pcg[key] = {}
            for key1 in prediction_from_scales[key]:
                mae[key][key1] = evaluate.mean_absolute_error(
                    predictions_dict[key][key1], dispL, maskL)
                err_pcg[key][key1] = evaluate.percentage_over_limit(
                    predictions_dict[key][key1], dispL, maskL, threshold)
    else:
        mae, err_pcg = None, None

    return mae, err_pcg, predictions_dict


def validate(which, form, merged, batch_size, stats, model_instance, initial_scale,
             scales, prediction_from_scales, device):
    assert which in ['val', 'test']
    assert form in ['crop', 'full']

    # create dataset
    form1 = form if form == 'crop' else 'full_im'
    data_feeder = preprocess.dataset(merged, which, form1, True)
    dataloader = torch.utils.data.DataLoader(data_feeder, batch_size,
                                             shuffle=True, num_workers=batch_size)

    # starting validation
    print("#### Starting validation on %s_%s after training epoch %d ####\n"
          % (which, form, stats['general']['cur_epoch']))

    epoch = stats['general']['cur_epoch']
    val_start_time = time.time()
    for key in prediction_from_scales.keys():
        for key1 in prediction_from_scales[key]:
            if str(key)+'_'+key1 not in stats[which+'_'+form]['mae'].keys():
                stats[which+'_'+form]['mae'][str(key)+'_'+key1] = []
            if str(key)+'_'+key1 not in stats[which+'_'+form]['pcg'].keys():
                stats[which+'_'+form]['pcg'][str(key)+'_'+key1] = []
                
            stats[which+'_'+form]['mae'][str(key)+'_'+key1].append([])
            stats[which+'_'+form]['pcg'][str(key)+'_'+key1].append([])

    for batch_idx, (imL, imR, dispL, maskL) in enumerate(dataloader):
        if maskL.sum() > 0:
            batch_start_time = time.time()
            imL = imL.cuda()
            imR = imR.cuda()
            dispL = dispL.cuda()
            maskL = maskL.cuda()

            mae, pcg, _ = inference(model_instance, initial_scale, scales,
                                    prediction_from_scales,
                                    device, imL, imR, dispL, maskL)
            time_passed = time.time() - batch_start_time

            # print results
            print('\n')
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key]:
                    print('val_on: %s_%s, iter: %d, %s_%s  MAE = %.3f , PCG = %2.f , time = %.2f' %
                          (which, form, batch_idx+1, key, key1, mae[key][key1],
                           pcg[key][key1], time_passed))

            # update training statistics
            for key in prediction_from_scales.keys():
                for key1 in prediction_from_scales[key]:
                    stats[which+'_' +
                          form]['mae'][str(key)+'_'+key1][epoch-1].append(mae[key][key1])
                    stats[which+'_' +
                          form]['pcg'][str(key)+'_'+key1][epoch-1].append(pcg[key][key1])

    # print after validation is finished
    print('\n#### Epoch %d validation on %s_%s results #### \n'
          % (epoch, which, form))
    print('time = %.3f' % (time.time() - val_start_time))
    for key in prediction_from_scales.keys():
        for key1 in prediction_from_scales[key]:
            print('%s_%s, μ(MAE) = %.3f' %
                  (key, key1, np.mean(stats[which+'_'+form]['mae'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, μ(PCG) = %.2f' %
                  (key, key1, np.mean(stats[which+'_'+form]['pcg'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, σ(MAE) = %.3f' %
                  (key, key1, np.std(stats[which+'_'+form]['mae'][str(key)+'_'+key1][epoch-1])))
            print('%s_%s, σ(PCG) = %.2f' %
                  (key, key1, np.std(stats[which+'_'+form]['pcg'][str(key)+'_'+key1][epoch-1])))
    print('\n')


def save_checkpoint(model_instance, optimizer, stats, save_to, epoch=None):
    if epoch is not None:
        savefilename = save_to + '/checkpoint_' + str(epoch) + '.tar'
    else:
        savefilename = save_to + '/checkpoint_' + \
            str(stats['general']['cur_epoch']) + '.tar'
    torch.save({
        'state_dict': model_instance.state_dict(),
        'optimizer': optimizer.state_dict(),
        'stats': stats
    }, savefilename)
