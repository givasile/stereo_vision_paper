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


cnn_name = 'multires_3d_less'

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
evaluate = importlib.import_module('evaluate')
preprocess = importlib.import_module('preprocess')


def convbn(in_ch, out_ch, kernel, stride, padding, dilation):
    conv = torch.nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                           dilation, bias=False)
    return torch.nn.Sequential(conv, torch.nn.BatchNorm2d(out_ch))


def downsample(in_ch, out_ch):
    conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                           dilation=1, bias=False)
    return torch.nn.Sequential(conv, torch.nn.BatchNorm2d(out_ch))


def convbn_3d(in_ch, out_ch, kernel, stride, padding, dilation):
    conv = torch.nn.Conv3d(in_ch, out_ch, kernel, stride,
                           padding, dilation, bias=False)
    return torch.nn.Sequential(conv, torch.nn.BatchNorm3d(out_ch))


class basic_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = convbn(32, 32, 3, 1, 1, 1)
        self.conv2 = convbn(32, 32, 3, 1, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sequence = torch.nn.Sequential(self.conv1, self.relu,
                                            self.conv2, self.relu)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        return out

class basic_layer_3d(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = convbn_3d(ch, ch, 3, 1, 1, 1)
        self.conv2 = convbn_3d(ch, ch, 3, 1, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sequence = torch.nn.Sequential(self.conv1, self.relu,
                                            self.conv2, self.relu)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        return out

class descriptors(torch.nn.Module):
    def __init__(self, n_bl):
        super().__init__()

        # first layer (kernel: 5x5, stride: 2)
        self.layer_1 = torch.nn.Sequential(convbn(3, 32, 5, 2, 2, 1),
                                           torch.nn.ReLU(inplace=True))

        # second layer (kernel: 5x5, stride: 2)
        self.layer_2 = torch.nn.Sequential(convbn(32, 32, 5, 2, 2, 1),
                                           torch.nn.ReLU(inplace=True))

        # Sequential of n_bl basic layers
        basic_layers = []
        for i in range(n_bl):
            basic_layers.append(basic_layer())
        self.sequence = torch.nn.Sequential(*basic_layers)

        # last layer with convolution only (+ bias)
        self.last_layer = torch.nn.Conv2d(32, 16, 3, 1, 1)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.sequence(out)
        out = self.last_layer(out)
        return out

def patch_comparison_volume(imL_d, imR_d, max_disp, is_training, device):
    b = imL_d.size()[0]
    ch = imL_d.size()[1]
    h = imL_d.size()[2]
    w = imL_d.size()[3]

    # initialize empty tensor
    vol = torch.zeros([b, ch*2, max_disp, h, w], dtype=torch.float32,
                      device=device, requires_grad=False)

    # fill with values
    vol[:, :ch, 0, :, :] = imL_d
    vol[:, ch:, 0, :, :] = imR_d
    for i in range(1, max_disp):
        vol[:, :ch, i, :, i:] = imL_d[:, :, :, i:]
        vol[:, ch:, i, :, i:] = imR_d[:, :, :, :-i]

    # assure tensor is contiguous
    if not vol.is_contiguous():
        vol = vol.contiguous()

    return vol

        # # 0
        # self.first_layer = torch.nn.Sequential(convbn_3d(64, 32, 3, 1, 1, 1),
        #                                        torch.nn.ReLU(inplace=True),
        #                                        basic_layer_3d(32))
                                               
        # # down -> 1
        # self.conv_stride_1 = torch.nn.Sequential(convbn_3d(32, 32, 3, 2, 1, 1),
        #                                          torch.nn.ReLU(
        #                                              inplace=True),
        #                                          convbn_3d(
        #                                              32, 32, 3, 1, 1, 1),
        #                                          torch.nn.ReLU(inplace=True))

        # # down -> 2        
        # self.conv_stride_2 = torch.nn.Sequential(convbn_3d(32, 64, 3, 2, 1, 1),
        #                                          torch.nn.ReLU(
        #                                              inplace=True),
        #                                          convbn_3d(
        #                                              64, 64, 3, 1, 1, 1),
        #                                          torch.nn.ReLU(inplace=True))

        # # same -> 2
        # self.conv_2 = basic_layer_3d(64)
        # self.to_out_2 = torch.nn.Sequential(convbn_3d(64, 32, 3, 1, 1, 1),
        #                                     torch.nn.ReLU(inplace=True))


        # # up -> 1
        # self.conv_up_1 = torch.nn.Sequential(torch.nn.ConvTranspose3d(
        #     64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #                                      torch.nn.BatchNorm3d(32),
        #                                      torch.nn.ReLU(inplace=False),
        #                                      torch.nn.Conv3d(32, 32, 3, 1, 1, 1, bias=False),
        #                                      torch.nn.BatchNorm3d(32),
        #                                      torch.nn.ReLU(inplace=False))

        # # up -> 0
        # self.conv_up_0 = torch.nn.Sequential(torch.nn.ConvTranspose3d(
        #     32, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #                                      torch.nn.BatchNorm3d(32),
        #                                      torch.nn.ReLU(inplace=False),
        #                                      torch.nn.Conv3d(32, 32, 3, 1, 1, 1, bias=False),
        #                                      torch.nn.BatchNorm3d(32),
        #                                      torch.nn.ReLU(inplace=False))
        
        # # to regression
        # self.to_regression = torch.nn.Sequential(basic_layer_3d(32),
        #                                          torch.nn.Conv3d(32, 1, 3, 1, 1, 1, bias=False))

class cnn_3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 0
        self.first_layer = torch.nn.Sequential(basic_layer_3d(32))
                                               
        # down -> 1
        self.conv_stride_1 = torch.nn.Sequential(convbn_3d(32, 32, 3, 2, 1, 1),
                                                 torch.nn.ReLU(
                                                     inplace=True),
                                                 convbn_3d(
                                                     32, 32, 3, 1, 1, 1),
                                                 torch.nn.ReLU(inplace=True))

        # down -> 2        
        self.conv_stride_2 = torch.nn.Sequential(convbn_3d(32, 32, 3, 2, 1, 1),
                                                 torch.nn.ReLU(
                                                     inplace=True),
                                                 convbn_3d(
                                                     32, 32, 3, 1, 1, 1),
                                                 torch.nn.ReLU(inplace=True))

        # same -> 2
        self.conv_2 = basic_layer_3d(32)


        # up -> 1
        self.conv_up_1 = torch.nn.Sequential(torch.nn.ConvTranspose3d(
            32, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                             torch.nn.BatchNorm3d(32),
                                             torch.nn.ReLU(inplace=False),
                                             torch.nn.Conv3d(32, 32, 3, 1, 1, 1, bias=False),
                                             torch.nn.BatchNorm3d(32),
                                             torch.nn.ReLU(inplace=False))

        # up -> 0
        self.conv_up_0 = torch.nn.Sequential(torch.nn.ConvTranspose3d(
            32, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                             torch.nn.BatchNorm3d(32),
                                             torch.nn.ReLU(inplace=False),
                                             torch.nn.Conv3d(32, 32, 3, 1, 1, 1, bias=False),
                                             torch.nn.BatchNorm3d(32),
                                             torch.nn.ReLU(inplace=False))
        
        # to regression
        self.to_regression = torch.nn.Sequential(basic_layer_3d(32),
                                                 torch.nn.Conv3d(32, 1, 3, 1, 1, 1, bias=False))
        
    def forward(self, x):
        
        # %4
        res_4 = self.first_layer(x)

        # %8
        res_8 = self.conv_stride_1(res_4)

        # %16
        out = self.conv_stride_2(res_8)
        out = self.conv_2(out)
        out_16 = self.to_regression(out)
        
        # %8
        out = self.conv_up_1(out)
        out += res_8
        out_8 = self.to_regression(out)
        
        # %4
        out = self.conv_up_0(out)
        out += res_4
        out_4 = self.to_regression(out)
        
        return out_4, out_8, out_16

def softargmax(vol, new_dimension, device):
    vol = torch.nn.functional.interpolate(vol, new_dimension, mode='trilinear', align_corners=True)
    
    # prepare vol
    vol = vol.squeeze(1)
    vol = torch.nn.functional.softmax(vol, 1)

    # prepare coeffs
    tmp = torch.linspace(0, vol.shape[1]-1, steps=vol.shape[1], requires_grad=False)
    tmp = tmp.reshape((1, vol.shape[1], 1, 1)).expand(
        (vol.shape[0], vol.shape[1], vol.shape[2], vol.shape[3]))

    tmp = tmp.contiguous().to(device)

    return torch.sum(tmp*vol, 1)


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.descriptor = descriptors(5)
        self.cnn_3D = cnn_3D()

    def forward(self, imL, imR, maxD, inspection, device):
        # extract features
        imL_d = self.descriptor(imL)
        imR_d = self.descriptor(imR)

        # matching
        cost_a = patch_comparison_volume(imL_d, imR_d, round(maxD/4), self.training, device)

        # 3D cnn
        cost_b_4, cost_b_8, cost_b_16  = self.cnn_3D(cost_a)

        # soft_argmax
        pred_4 = softargmax(cost_b_4, [maxD, imL.shape[2], imL.shape[3]], device)
        pred_8 = softargmax(cost_b_8, [maxD, imL.shape[2], imL.shape[3]], device)
        pred_16 = softargmax(cost_b_16, [maxD, imL.shape[2], imL.shape[3]], device)

        if inspection:
            return pred_4, pred_8, pred_16, imL_d, imR_d, cost_a, cost_b_4, cost_b_8, cost_b_16
        else:
            return pred_4, pred_8, pred_16
        
def training_step(model_instance, optimizer, loss_from_scales_in_training, imL, imR,
                  dispL, maskL, maxD, device):

    if maskL.sum() > 0:
        # forward, backward and update
        model_instance.train()
        optimizer.zero_grad()

        pred_4, pred_8, pred_16 = model_instance(imL, imR, maxD, False, device)

        loss_at_scales = []
        loss_at_scales.append(torch.nn.functional.smooth_l1_loss(
                pred_4[maskL], dispL[maskL]))
        loss_at_scales.append(torch.nn.functional.smooth_l1_loss(
                pred_8[maskL], dispL[maskL]))
        loss_at_scales.append(torch.nn.functional.smooth_l1_loss(
                pred_16[maskL], dispL[maskL]))
        
        loss = 0
        for i, l in enumerate(loss_at_scales):
            loss += l * loss_from_scales_in_training[i]

        loss.backward()
        optimizer.step()

        # statistics
        pred_4 = pred_4.detach()
        pred_8 = pred_8.detach()
        pred_16 = pred_16.detach()

        mae = []
        mae.append(evaluate.mean_absolute_error(pred_4, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_8, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_16, dispL, maskL))

        pcg = []
        threshold = 3
        pcg.append(evaluate.percentage_over_limit(pred_4, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_8, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_16, dispL, maskL, threshold))
        
        return mae, pcg, loss_at_scales, loss
    else:
        return None, None, None, None


def training_epoch(merged, batch_size, stats, model_instance, optimizer, maxD,
                   loss_from_scales_in_training, device):
    # prepare dataset
    data_feeder = preprocess.dataset(merged, 'train', 'crop', True)
    dataloader = torch.utils.data.DataLoader(data_feeder, batch_size=batch_size,
                                             shuffle=True, num_workers=batch_size)
    epoch = stats['general']['cur_epoch']
    print('Starting epoch %d of training' % (epoch))
    epoch_start_time = time.time()

    # add new epoch on statistics
    keys1 = ['mae', 'pcg', 'loss_at_scales']
    keys2 = ['_4', '_8', '_16']
    stats['train']['loss'].append([])
    for key1 in keys1:
        for key2 in keys2:
            stats['train'][key1][key2].append([])

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

            mae, pcg, loss_at_scales, loss = training_step(model_instance, optimizer,
                                                           loss_from_scales_in_training,
                                                           imL, imR, dispL, maskL, maxD, device)

            # print results
            print('\n')
            print('Iter %d: LOSS = %.3f, time = %.2f' %
                  (batch_idx+1, loss, time.time() - batch_start_time))

            # update training statistics
            keys = ['_4', '_8', '_16']
            stats['train']['loss'][epoch-1].append(loss.item())
            for i, key in enumerate(keys):
                stats['train']['mae'][key][epoch-1].append(mae[i])
                stats['train']['pcg'][key][epoch-1].append(pcg[i])
                stats['train']['loss_at_scales'][key][epoch-1].append(loss_at_scales[i].item())

            stats['general']['cur_step'] += 1

    # print results after training epoch
    print('\n#### Results on training epoch %d #### \n' % (epoch))
    print('time = %.3f' % (time.time() - epoch_start_time))
    keys = ['_4', '_8', '_16']
    for key in keys:
        print('μ(MAE%s) = %.3f' %
              (key, np.mean(stats['train']['mae'][key][epoch-1])))
        print('μ(PCG%s) = %.2f' %
              (key, np.mean(stats['train']['pcg'][key][epoch-1])))
        print('σ(MAE%s) = %.3f' %
              (key, np.std(stats['train']['mae'][key][epoch-1])))
        print('σ(PCG%s) = %.2f' %
              (key, np.std(stats['train']['pcg'][key][epoch-1])))
    print('\n')


def inspection(model_instance, device, mode, imL, imR, dispL, maskL, maxD):
    if maskL.sum() > 0:

        if mode == 'train':
            model_instance.train()
        elif mode == 'eval':
            model_instance.eval()

        with torch.no_grad():
            pred_4, pred_8, pred_16, imL_d, imR_d, cost_a, cost_b_4, cost_b_8, cost_b_16 = model_instance(
                imL, imR, maxD, True, device)

        pred_4 = pred_4.detach()
        pred_8 = pred_8.detach()
        pred_16 = pred_16.detach()

        # statistics
        mae = []
        mae.append(evaluate.mean_absolute_error(pred_4, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_8, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_16, dispL, maskL))

        pcg = []
        threshold = 3
        pcg.append(evaluate.percentage_over_limit(pred_4, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_8, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_16, dispL, maskL, threshold))
        return [mae, pcg, imL_d, imR_d, pred_4, pred_8, pred_16,
                imL_d, imR_d, cost_a, cost_b_4, cost_b_8, cost_b_16]
    else:
        return None




def inference(model_instance, device, imL, imR, dispL, maskL, maxD):
    if maskL.sum() > 0:
        model_instance.eval()

        with torch.no_grad():
            pred_4, pred_8, pred_16 = model_instance(imL, imR, maxD, False, device)

        pred_4 = pred_4.detach()
        pred_8 = pred_8.detach()
        pred_16 = pred_16.detach()

        # statistics
        mae = []
        mae.append(evaluate.mean_absolute_error(pred_4, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_8, dispL, maskL))
        mae.append(evaluate.mean_absolute_error(pred_16, dispL, maskL))

        pcg = []
        threshold = 3
        pcg.append(evaluate.percentage_over_limit(pred_4, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_8, dispL, maskL, threshold))
        pcg.append(evaluate.percentage_over_limit(pred_16, dispL, maskL, threshold))
        return [mae, pcg]
    else:
        return None


def validate(which, form, merged, batch_size, stats, model_instance, device, maxD):
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
    
    keys1 = ['mae', 'pcg']
    keys2 = ['_4', '_8', '_16']
    for key1 in keys1:
        for key2 in keys2:
            stats[which+'_'+form][key1][key2].append([])

    for batch_idx, (imL, imR, dispL, maskL) in enumerate(dataloader):
        if maskL.sum() > 0:
            batch_start_time = time.time()
            imL = imL.cuda()
            imR = imR.cuda()
            dispL = dispL.cuda()
            maskL = maskL.cuda()

            max_disp = 192

            mae, pcg = inference(model_instance, device, imL, imR, dispL, maskL, maxD)
            time_passed = time.time() - batch_start_time

            # print results
            print('\n')
            print('val_on: %s_%s, iter: %d, MAE = %.3f , PCG = %2.f , time = %.2f' %
                  (which, form, batch_idx+1, mae[0], pcg[0], time_passed))

            # update training statistics
            keys = ['_4', '_8', '_16']
            for i, key in enumerate(keys):
                stats[which + '_' + form]['mae'][key][epoch-1].append(mae[i])
                stats[which + '_' + form]['pcg'][key][epoch-1].append(pcg[i])

    # print after validation is finished
    print('\n#### Results on training epoch %d #### \n' % (epoch))
    print('time = %.3f' % (time.time() - val_start_time))
    keys = ['_4', '_8', '_16']
    for key in keys:
        print('μ(MAE%s) = %.3f' %
              (key, np.mean(stats[which + '_' + form]['mae'][key][epoch-1])))
        print('μ(PCG%s) = %.2f' %
              (key, np.mean(stats[which + '_' + form]['pcg'][key][epoch-1])))
        print('σ(MAE%s) = %.3f' %
              (key, np.std(stats[which + '_' + form]['mae'][key][epoch-1])))
        print('σ(PCG%s) = %.2f' %
              (key, np.std(stats[which + '_' + form]['pcg'][key][epoch-1])))
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
