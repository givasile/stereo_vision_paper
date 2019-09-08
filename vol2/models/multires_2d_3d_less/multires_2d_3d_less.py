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


cnn_name = 'multires_2d_3d_less'

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
    return torch.nn.Sequential(conv,
                               torch.nn.BatchNorm2d(out_ch),
                               torch.nn.ReLU(inplace=False))


def convbn_3d(in_ch, out_ch, kernel, stride, padding, dilation):
    conv = torch.nn.Conv3d(in_ch, out_ch, kernel, stride,
                           padding, dilation, bias=False)
    return torch.nn.Sequential(conv,
                               torch.nn.BatchNorm3d(out_ch))

def conv_3d(in_ch, out_ch, kernel, stride, padding, dilation):
    conv = torch.nn.Conv3d(in_ch, out_ch, kernel, stride,
                           padding, dilation, bias=False)
    return conv


class basic_layer_3d(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = convbn_3d(ch, ch, 3, 1, 1, 1)
        # self.conv2 = convbn_3d(ch, ch, 3, 1, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sequence = torch.nn.Sequential(self.conv1, self.relu)
                                            # self.conv2, self.relu)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        return out

class basic_layer(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = convbn(ch, ch, 3, 1, 1, 1)
        # self.conv2 = convbn(ch, ch, 3, 1, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sequence = torch.nn.Sequential(self.conv1, self.relu)
                                            # self.conv2, self.relu)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        return out


class descriptor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(convbn(3, 16, 3, 1, 1, 1),
                                         torch.nn.ReLU(inplace=True))
        # first seq
        self.layer_1_1 = torch.nn.Sequential(convbn(16, 16, 3, 1, 1, 1),
                                             torch.nn.ReLU(inplace=True),
                                             # convbn(16, 16, 3, 1, 1, 1),
                                             # torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(16, 16, 3, 1, 1, 1))
        self.layer_1_2 = torch.nn.Sequential(downsample(16, 16),
                                             # basic_layer(16),
                                             basic_layer(16))

        # second seq
        self.layer_2_1 = torch.nn.Sequential(convbn(16, 16, 3, 1, 1, 1),
                                             torch.nn.ReLU(inplace=True),
                                             # convbn(16, 16, 3, 1, 1, 1),
                                             # torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(16, 16, 3, 1, 1, 1))
        self.layer_2_2 = torch.nn.Sequential(downsample(16, 16),
                                             # basic_layer(16),
                                             basic_layer(16))

        # third seq
        self.layer_3_1 = torch.nn.Sequential(convbn(16, 16, 3, 1, 1, 1),
                                             torch.nn.ReLU(inplace=True),
                                             # convbn(16, 16, 3, 1, 1, 1),
                                             # torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(16, 16, 3, 1, 1, 1))
        self.layer_3_2 = torch.nn.Sequential(downsample(16, 16),
                                             # basic_layer(16),
                                             basic_layer(16))
        # fourth seq 
        self.layer_4_1 = torch.nn.Sequential(convbn(16, 16, 3, 1, 1, 1),
                                             torch.nn.ReLU(inplace=True),
                                             # convbn(16, 16, 3, 1, 1, 1),
                                             # torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(16, 16, 3, 1, 1, 1))
        self.layer_4_2 = torch.nn.Sequential(downsample(16, 16),
                                             # basic_layer(16),
                                             basic_layer(16))

        # fifth seq
        self.layer_5_1 = torch.nn.Sequential(basic_layer(16))

        self.stride_4_4 = torch.nn.AvgPool2d(4, 4)
        self.stride_2_2 = torch.nn.AvgPool2d(2, 2)

    def forward(self, x):

        # first layer
        out = self.conv1(x)

        # block1
        resol1 = self.layer_1_1(out)
        resol1 = self.stride_4_4(resol1)
        out = self.layer_1_2(out)

        # block2
        resol2 = self.layer_2_1(out)
        resol2 = self.stride_2_2(resol2)
        out = self.layer_2_2(out)

        # block3
        resol3 = self.layer_3_1(out)
        out = self.layer_3_2(out)

        # block4
        resol4 = self.layer_4_1(out)
        out = self.layer_4_2(out)

        # block5
        resol5 = self.layer_5_1(out)

        return resol1, resol2, resol3, resol4, resol5


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




class cnn_3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # res5
        self.res5_1 = torch.nn.Sequential(conv_3d(32, 64, 3, 1, 1, 1),
                                          torch.nn.ReLU(inplace=False),
                                          basic_layer_3d(64),
                                          torch.nn.ConvTranspose3d(
                                              64, 32, kernel_size=3, padding=1,
                                              output_padding=1, stride=2, bias=False))

        # res4
        self.res4_1 = torch.nn.Sequential(conv_3d(64, 32, 3, 1, 1, 1),
                                          torch.nn.BatchNorm3d(32),
                                          torch.nn.ReLU(inplace=False),
                                          basic_layer_3d(32),
                                          torch.nn.ConvTranspose3d(
                                              32, 32, kernel_size=3, padding=1,
                                              output_padding=1, stride=2, bias=False))

        # res3
        self.res3_1 = torch.nn.Sequential(conv_3d(64, 32, 3, 1, 1, 1),
                                          torch.nn.BatchNorm3d(32),
                                          torch.nn.ReLU(inplace=False),
                                          basic_layer_3d(32),
                                          conv_3d(32, 32, 3, 1, 1, 1))
        
        # res2
        self.res2_1 = torch.nn.Sequential(conv_3d(64, 32, 3, 1, 1, 1),
                                          torch.nn.BatchNorm3d(32),
                                          # basic_layer_3d(32),
                                          torch.nn.ReLU(inplace=False),
                                          conv_3d(32, 32, 3, 1, 1, 1))


        # res1
        self.res1_1 = torch.nn.Sequential(conv_3d(64, 32, 3, 1, 1, 1),
                                          torch.nn.BatchNorm3d(32),
                                          # basic_layer_3d(32),
                                          torch.nn.ReLU(inplace=False),
                                          conv_3d(32, 32, 3, 1, 1, 1))
 

        # classify
        self.classify_32 = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                               torch.nn.ReLU(inplace=True),
                                               # convbn_3d(32, 32, 3, 1, 1, 1),
                                               # torch.nn.ReLU(inplace=True),
                                               torch.nn.Conv3d(32, 1, 3, 1, 1,
                                                               bias=False))
        
        

    def forward(self, cost):

        # res 4
        cost4_b = self.res5_1(cost[4])
        cost4 = torch.cat((cost4_b, cost[3]), 1)
        
        # res 3
        cost3_b = self.res4_1(cost4)
        cost3 = torch.cat((cost3_b, cost[2]), 1)

        # res 2
        cost2_b = self.res3_1(cost3)
        cost2 = torch.cat((cost2_b, cost[1]), 1)
        
        # res 1
        cost1_b = self.res2_1(cost2)
        cost1 = torch.cat((cost1_b, cost[0]), 1)

        # res 0
        cost1_a = self.res1_1(cost1)
        
        # for prediction
        return [self.classify_32(cost1_a), self.classify_32(cost1_b), self.classify_32(cost2_b),
                self.classify_32(cost3_b), self.classify_32(cost4_b)]


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

        self.descriptor = descriptor()
        self.cnn_3D = cnn_3D()

    def forward(self, imL, imR, maxD, inspection, device):
        # extract features
        imL_d = self.descriptor(imL)
        imR_d = self.descriptor(imR)

        # matching
        mapping = [round(maxD/4), round(maxD/4), round(maxD/4), round(maxD/8), round(maxD/16)]
        cost_a = []
        for i in range(5):
            cost_a.append(patch_comparison_volume(imL_d[i], imR_d[i], mapping[i], self.training, device))

        # 3D cnn
        cost_b = self.cnn_3D(cost_a)
        # softargmax
        pred = []
        for i in range(5):
            pred.append(softargmax(cost_b[i], [maxD, imL.shape[2], imL.shape[3]], device))

        if inspection:
            return pred, imL_d, imR_d, cost_a, cost_b
        else:
            return pred


def training_step(model_instance, optimizer, loss_from_scales_in_training, imL, imR,
                  dispL, maskL, maxD, device):

    if maskL.sum() > 0:
        # forward, backward and update
        model_instance.train()
        optimizer.zero_grad()
        pred = model_instance(imL, imR, maxD, False, device)

        loss_at_scales = []
        for i in range(5):
            loss_at_scales.append(torch.nn.functional.smooth_l1_loss(
                pred[i][maskL], dispL[maskL]))

        loss = 0
        for i, l in enumerate(loss_at_scales):
            loss += l * loss_from_scales_in_training[i]

        loss.backward()
        optimizer.step()

        # statistics
        for i in range(5):
            pred[i] = pred[i].detach()

        mae = []
        for i in range(5):
            mae.append(evaluate.mean_absolute_error(pred[i], dispL, maskL))

        pcg = []
        threshold = 3
        for i in range(5):
            pcg.append(evaluate.percentage_over_limit(pred[i], dispL, maskL, threshold))
        
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
    keys2 = ['_1', '_2', '_4', '_8', '_16']
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
            print('Iter %d: LOSS = %.3f, MAE_1 = %.3f, MAE_2 = %.3f, MAE_4 = %.3f, MAE_8 = %.3f, MAE_16 = %.3f, time = %.2f' %
                  (batch_idx+1, loss, mae[0], mae[1], mae[2], mae[3], mae[4], time.time() - batch_start_time))

            # update training statistics
            keys = ['_1', '_2', '_4', '_8', '_16']
            stats['train']['loss'][epoch-1].append(loss.item())
            for i, key in enumerate(keys):
                stats['train']['mae'][key][epoch-1].append(mae[i])
                stats['train']['pcg'][key][epoch-1].append(pcg[i])
                stats['train']['loss_at_scales'][key][epoch-1].append(loss_at_scales[i].item())

            stats['general']['cur_step'] += 1

    # print results after training epoch
    print('\n#### Results on training epoch %d #### \n' % (epoch))
    print('time = %.3f' % (time.time() - epoch_start_time))
    keys = ['_1', '_2', '_4', '_8', '_16']
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
            pred, imL_d, imR_d, cost_a, cost_b = model_instance(
                imL, imR, maxD, True, device)

        for i in range(5):
            pred[i] = pred[i].detach()

        mae = []
        for i in range(5):
            mae.append(evaluate.mean_absolute_error(pred[i], dispL, maskL))

        pcg = []
        threshold = 3
        for i in range(5):
            pcg.append(evaluate.percentage_over_limit(pred[i], dispL, maskL, threshold))

        return [mae, pcg, pred, imL_d, imR_d, cost_a, cost_b]
    else:
        return None


def inference(model_instance, device, imL, imR, dispL, maskL, maxD):
    if maskL.sum() > 0:
        model_instance.eval()

        with torch.no_grad():
            pred = model_instance(imL, imR, maxD, False, device)

        for i in range(5):
            pred[i] = pred[i].detach()

        mae = []
        for i in range(5):
            mae.append(evaluate.mean_absolute_error(pred[i], dispL, maskL))

        pcg = []
        threshold = 3
        for i in range(5):
            pcg.append(evaluate.percentage_over_limit(pred[i], dispL, maskL, threshold))

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
    keys2 = ['_1', '_2', '_4', '_8', '_16']
    for key1 in keys1:
        for key2 in keys2:
            stats[which+'_'+form][key1][key2].append([])

    imag = 0
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
            keys = ['_1', '_2', '_4', '_8', '_16']
            for i, key in enumerate(keys):
                stats[which + '_' + form]['mae'][key][epoch-1].append(mae[i])
                stats[which + '_' + form]['pcg'][key][epoch-1].append(pcg[i])
        else:
            imag += 1
            print('aek'+ str(imag))

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
