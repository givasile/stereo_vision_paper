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


cnn_name = 'multires_2d'

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
    def __init__(self, ch):
        super().__init__()
        self.conv1 = convbn(ch, ch, 3, 1, 1, 1)
        self.conv2 = convbn(ch, ch, 3, 1, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sequence = torch.nn.Sequential(self.conv1, self.relu,
                                            self.conv2, self.relu)

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
        self.layer_1_1 = basic_layer(16)
        self.layer_1_2 = torch.nn.Conv2d(16, 16, 3, 1, 1, 1)
        self.layer_1_3 = downsample(16, 16)

        # second seq
        self.layer_2_1 = basic_layer(16)
        self.layer_2_2 = torch.nn.Conv2d(16, 16, 3, 1, 1, 1)
        self.layer_2_3 = downsample(16, 32)

        # third seq
        self.layer_3_1 = basic_layer(32)
        self.layer_3_2 = torch.nn.Conv2d(32, 16, 3, 1, 1, 1)
        self.layer_3_3 = downsample(32, 64)

        # fourth seq
        self.layer_4_1 = basic_layer(64)
        self.layer_4_2 = torch.nn.Conv2d(64, 16, 3, 1, 1, 1)
        self.layer_4_3 = downsample(64, 64)

        # fifth seq
        self.layer_5_1 = basic_layer(64)
        self.layer_5_2 = torch.nn.Conv2d(64, 16, 3, 1, 1, 1)

        self.stride_4_4 = torch.nn.AvgPool2d(4, 4)
        self.stride_2_2 = torch.nn.AvgPool2d(2, 2)

    def forward(self, x, rand):
        # first layer
        out = self.conv1(x)

        # block1
        out = self.layer_1_1(out)
        resol1 = self.layer_1_2(out)
        out = self.layer_1_3(out)

        # block2
        out = self.layer_2_1(out)
        resol2 = self.layer_2_2(out)
        out = self.layer_2_3(out)

        # block3
        out = self.layer_3_1(out)
        resol3 = self.layer_3_2(out)
        out = self.layer_3_3(out)

        # block4
        out = self.layer_4_1(out)
        resol4 = self.layer_4_2(out)
        out = self.layer_4_3(out)

        # block5
        out = self.layer_5_1(out)
        resol5 = self.layer_5_2(out)

        # downsample
        resol1 = torch.nn.functional.avg_pool2d(resol1, 4, 4)
        resol2 = torch.nn.functional.avg_pool2d(resol2, 2, 2)
        
        # upsample
        resol4 = torch.nn.functional.interpolate(
            resol4, (int(x.size(2)/4), int(x.size(3)/4)), mode='bilinear',
            align_corners=True)
        resol5 = torch.nn.functional.interpolate(
            resol5, (int(x.size(2)/4), int(x.size(3)/4)), mode='bilinear',
            align_corners=True)

        # concatenate
        list_of_scales = [resol1, resol2, resol3, resol4, resol5]
        for i in range(5):
            if rand[i]:
                list_of_scales[i] = torch.zeros_like(list_of_scales[i], requires_grad=False)

        descriptor = torch.cat(list_of_scales, 1)
        return descriptor


def patch_comparison_volume(imL_d, imR_d, max_disp, is_training, device):
    b = imL_d.size()[0]
    ch = imL_d.size()[1]
    h = imL_d.size()[2]
    w = imL_d.size()[3]

    # initialize empty tensor
    vol = torch.zeros([b, ch*2, max_disp+1, h, w], dtype=torch.float32,
                      device=device, requires_grad=False)

    # fill with values
    vol[:, :ch, 0, :, :] = imL_d
    vol[:, ch:, 0, :, :] = imR_d
    for i in range(1, max_disp + 1):
        vol[:, :ch, i, :, i:] = imL_d[:, :, :, i:]
        vol[:, ch:, i, :, i:] = imR_d[:, :, :, :-i]

    # assure tensor is contiguous
    if not vol.is_contiguous():
        vol = vol.contiguous()

    return vol




class cnn_3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dres0_0 = torch.nn.Sequential(convbn_3d(160, 128, 3, 1, 1, 1),
                                           torch.nn.ReLU(inplace=True),
                                           convbn_3d(128, 64, 3, 1, 1, 1),
                                           torch.nn.ReLU(inplace=True))

        self.dres0_1 = torch.nn.Sequential(convbn_3d(64, 64, 3, 1, 1, 1),
                                           torch.nn.ReLU(inplace=True),
                                           convbn_3d(64, 32, 3, 1, 1, 1),
                                           torch.nn.ReLU(inplace=True))

        self.dres1 = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                         torch.nn.ReLU(inplace=True),
                                         convbn_3d(32, 32, 3, 1, 1, 1))

        self.dres2 = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                         torch.nn.ReLU(inplace=True),
                                         convbn_3d(32, 32, 3, 1, 1, 1))

        self.dres3 = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                         torch.nn.ReLU(inplace=True),
                                         convbn_3d(32, 32, 3, 1, 1, 1))

        self.dres4 = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                         torch.nn.ReLU(inplace=True),
                                         convbn_3d(32, 32, 3, 1, 1, 1))

        self.classify = torch.nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Conv3d(32, 1, 3, 1, 1,
                                                            bias=False))

    def forward(self, cost):
        # process cost volume
        cost0 = self.dres0_0(cost)
        cost0 = self.dres0_1(cost0)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0
        cost = self.classify(cost0)

        return cost

# def softargmax(vol, coeff, device):
#     # prepare vol
#     vol = vol.squeeze(1)
#     vol = torch.nn.functional.softmax(vol, 1)

#     # prepare coeffs
#     tmp = torch.linspace(
#         0, vol.shape[1]-1, steps=vol.shape[1], requires_grad=False) * coeff
#     tmp = tmp.reshape((1, vol.shape[1], 1, 1)).expand(
#         (vol.shape[0], vol.shape[1], vol.shape[2], vol.shape[3]))

#     tmp = tmp.contiguous().to(device)

#     return torch.sum(tmp*vol, 1)


# def upsample_disparity_map(x, new_dimension):
#     x = x.unsqueeze(1)
#     x = torch.nn.functional.interpolate(
#         x, [new_dimension[0], new_dimension[1]], mode='bilinear',
#         align_corners=True)
#     return x.squeeze(1)

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
        # randomize input
        if self.training:
            rand = torch.rand(5) > 0.6
        else:
            rand = torch.zeros(5)

        # extract features
        imL_d = self.descriptor(imL, rand)
        imR_d = self.descriptor(imR, rand)

        # matching
        cost_a = patch_comparison_volume(imL_d, imR_d, round(maxD/4), self.training, device)

        # 3D cnn
        cost_b = self.cnn_3D(cost_a)

        # softargmax
        # tmp = softargmax(cost_b, round(imL.shape[2]/float(cost_b.shape[3])), device)
        # pred = upsample_disparity_map(tmp, imL.shape[2:])
        
        pred = softargmax(cost_b, [maxD, imL.shape[2], imL.shape[3]], device)

        if inspection:
            return pred, imL_d, imR_d, cost_a, cost_b
        else:
            return pred


def training_step(model_instance, optimizer, imL, imR, dispL, maskL, maxD, device):

    if maskL.sum() > 0:
        # forward, backward and update
        model_instance.train()
        optimizer.zero_grad()
        pred = model_instance(imL, imR, maxD, False, device)
        loss = torch.nn.functional.smooth_l1_loss(pred[maskL], dispL[maskL])
        loss.backward()
        optimizer.step()

        # statistics
        pred = pred.detach()
        threshold = 3
        mae = evaluate.mean_absolute_error(pred, dispL, maskL)
        std_ae = evaluate.std_absolute_error(pred, dispL, maskL)
        pcg = evaluate.percentage_over_limit(pred, dispL, maskL, threshold)
        return mae, std_ae, pcg, loss
    else:
        return None, None, None, None
    

def training_epoch(merged, batch_size, stats, model_instance, optimizer, maxD, device):
    # prepare dataset
    data_feeder = preprocess.dataset(merged, 'train', 'crop', True)
    dataloader = torch.utils.data.DataLoader(data_feeder, batch_size=batch_size,
                                             shuffle=True, num_workers=batch_size)
    epoch = stats['general']['cur_epoch']
    print('Starting epoch %d of training' % (epoch))
    epoch_start_time = time.time()

    # add new epoch on statistics
    stats['train']['loss'].append([])
    stats['train']['mae'].append([])
    stats['train']['std_ae'].append([])
    stats['train']['pcg'].append([])

    # for every batch
    for batch_idx, (imL, imR, dispL, maskL) in enumerate(dataloader):
        if maskL.sum() > 0:
            batch_start_time = time.time()
            
            if device == 'cuda':
                imL = imL.cuda()
                imR = imR.cuda()
                dispL = dispL.cuda()
                maskL = maskL.cuda()

            mae, std_ae, pcg, loss = training_step(model_instance, optimizer, imL, imR,
                                                   dispL, maskL, maxD, device)

            # print results
            print('\n')
            print('Iter %d: LOSS = %.3f, time = %.2f' %
                  (batch_idx+1, loss, time.time() - batch_start_time))

            # update training statistics
            stats['train']['loss'][epoch-1].append(loss.detach().item())
            stats['train']['mae'][epoch-1].append(mae)
            stats['train']['std_ae'][epoch-1].append(std_ae)
            stats['train']['pcg'][epoch-1].append(pcg)
            stats['general']['cur_step'] += 1

    # print results after training epoch
    print('\n#### Results on training epoch %d #### \n' % (epoch))
    print('time = %.3f' % (time.time() - epoch_start_time))

    print('μ(MAE) = %.3f' %(np.mean(stats['train']['mae'][epoch-1])))
    print('μ(std_AE) = %.3f' %(np.mean(stats['train']['std_ae'][epoch-1])))
    print('μ(PCG) = %.2f' %(np.mean(stats['train']['pcg'][epoch-1])))
    print('σ(MAE) = %.3f' %(np.std(stats['train']['mae'][epoch-1])))
    print('σ(std_AE) = %.3f' %(np.std(stats['train']['std_ae'][epoch-1])))
    print('σ(PCG) = %.2f' %(np.std(stats['train']['pcg'][epoch-1])))
    print('\n')


def inspection(model_instance, device, mode, imL, imR, dispL, maskL, maxD):
    if mode == 'train':
        model_instance.train()
    elif mode == 'eval':
        model_instance.eval()

    if maskL.sum() > 0:
        with torch.no_grad():
            pred, imL_d, imR_d, cost_a, cost_b = model_instance(imL, imR, maxD, True, device)
        pred = pred.detach()

        threshold = 3
        mae = evaluate.mean_absolute_error(pred, dispL, maskL)
        std_ae = evaluate.std_absolute_error(pred, dispL, maskL)
        pcg = evaluate.percentage_over_limit(pred, dispL, maskL, threshold)
        return [pred, imL_d, imR_d, cost_a, cost_b, mae, std_ae, pcg]
    else:
        return None


def inference(model_instance, device, imL, imR, dispL, maskL, maxD):
    model_instance.eval()

    if maskL.sum() > 0:
        with torch.no_grad():
            pred  = model_instance(imL, imR, maxD, False, device)

        pred = pred.detach()

        threshold = 3
        mae = evaluate.mean_absolute_error(pred, dispL, maskL)
        std_ae = evaluate.std_absolute_error(pred, dispL, maskL)
        pcg = evaluate.percentage_over_limit(pred, dispL, maskL, threshold)
        return [pred, mae, std_ae, pcg]
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
    stats[which+'_'+form]['mae'].append([])
    stats[which+'_'+form]['std_ae'].append([])
    stats[which+'_'+form]['pcg'].append([])

    for batch_idx, (imL, imR, dispL, maskL) in enumerate(dataloader):
        if maskL.sum() > 0:
            batch_start_time = time.time()
            if device == 'cuda':
                imL = imL.cuda()
                imR = imR.cuda()
                dispL = dispL.cuda()
                maskL = maskL.cuda()

            pred, mae, std_ae, pcg = inference(model_instance, device, imL, imR, dispL, maskL, maxD)
            time_passed = time.time() - batch_start_time

            # print results
            print('\n')
            print('val_on: %s_%s, iter: %d, MAE = %.3f , PCG = %2.f , time = %.2f' %
                  (which, form, batch_idx+1, mae, pcg, time_passed))

            # update training statistics
            stats[which+'_' +form]['mae'][epoch-1].append(mae)
            stats[which+'_' +form]['std_ae'][epoch-1].append(std_ae)
            stats[which+'_' +form]['pcg'][epoch-1].append(pcg)

    # print after validation is finished
    print('\n#### Epoch %d validation on %s_%s results #### \n'
          % (epoch, which, form))
    print('time = %.3f' % (time.time() - val_start_time))
    print('μ(MAE) = %.3f' %(np.mean(stats[which+'_'+form]['mae'][epoch-1])))
    print('μ(std_AE) = %.3f' %(np.mean(stats[which+'_'+form]['std_ae'][epoch-1])))
    print('μ(PCG) = %.2f' %(np.mean(stats[which+'_'+form]['pcg'][epoch-1])))
    print('σ(MAE) = %.3f' %(np.std(stats[which+'_'+form]['mae'][epoch-1])))
    print('σ(std_AE) = %.3f' %(np.std(stats[which+'_'+form]['std_ae'][epoch-1])))
    print('σ(PCG) = %.2f' %(np.std(stats[which+'_'+form]['pcg'][epoch-1])))
    print('\n')

    
def init_weights(mod):
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.fill_(1)
        elif isinstance(m, torch.nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * \
                m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.fill_(1)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


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
