import sys
import configparser
import torch
import os
import math
import numpy as np
import scipy.ndimage.filters as fil

# import config file
conf_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
conf_path += '/conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

# torch.manual_seed(5)

def gaussian_filter(imL, new_dimension, device):
    channels = imL.shape[1]
    dim = 2 # nof dimensions (=2, spatial data)
    downscale = imL.shape[2]/new_dimension[0]*0.5 + imL.shape[3]/new_dimension[1]*0.5
    sigma = downscale / 3.0
    radius = int(4.0*sigma + 0.5)
    kernel_size = 2*radius + 1

    # method
    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32)
         for size in kernel_size
         ]
    )

    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp((-((mgrid - mean) / (std)) ** 2)/2)
        
    kernel = kernel / torch.sum(kernel)

    # repeat along input's channels
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    
    if device == 'cuda':
        kernel = kernel.cuda()
    imL = torch.nn.functional.pad(imL, (radius, radius, radius, radius), 'reflect')
    imL = torch.nn.functional.conv2d(imL, weight = kernel, groups = channels)
    return imL


# f1: 2d downsampling
def downsampling_2d(x, new_dimension, device):
    
    # x = gaussian_filter(x, new_dimension, device)
    x = torch.nn.functional.interpolate(
        x, [new_dimension[0], new_dimension[1]], mode='bilinear',
        align_corners=False)
    return x


# helper module
class residual_2d_module(torch.nn.Module):
    # bn + relu + conv2d strategy
    def __init__(self, ch):
        super().__init__()
        self.seq1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch, ch, 3, 1, 1, 1, bias=False)
        )

        self.seq2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch, ch, 3, 1, 1, 1, bias=False))

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)
        out += x
        return out


# f2: 2d cnn between scales
# class cnn_2d_between_scales(torch.nn.Module):
#     def __init__(self, ch, num_of_residuals):
#         super().__init__()

#         layers = []
#         for i in range(num_of_residuals):
#             layers.append(residual_2d_module(ch))
#         self.seq = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.seq(x)


# f3: 2d cnn to compute comparable features
class cnn_2d_for_comparable_features(torch.nn.Module):
    def __init__(self, ch, num_of_residuals):
        super().__init__()

        layers = []
        for i in range(num_of_residuals):
            layers.append(residual_2d_module(ch))
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# 2d module: downsampling + f3
class module_2d(torch.nn.Module):
    def __init__(self, ch, num_of_residuals_on_f2,
                 num_of_residuals_on_f3):
        super().__init__()

        # self.f2 = cnn_2d_between_scales(ch, num_of_residuals_on_f2)
        self.f3 = cnn_2d_for_comparable_features(ch, num_of_residuals_on_f3)

    def forward(self, x, new_dimension, extract, device):
        out = downsampling_2d(x, new_dimension, device)
        # out = self.f2(out)
        if extract:
            out1 = self.f3(out)
        else:
            out1 = None
        return out, out1


class descriptors_extraction(torch.nn.Module):
    def __init__(self, ch, num_of_residuals_on_f2,
                 num_of_residuals_on_f3):
        super().__init__()

        # first layer
        self.first_2d_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, 1, 1, 1, bias=True)
        )

        # basic_module
        self.module_2d_1 = module_2d(ch, num_of_residuals_on_f2,
                                     num_of_residuals_on_f3)
        self.module_2d_2 = module_2d(ch, num_of_residuals_on_f2,
                                     num_of_residuals_on_f3)
        self.module_2d_3 = module_2d(ch, num_of_residuals_on_f2,
                                     num_of_residuals_on_f3)
        self.module_2d_4 = module_2d(ch, num_of_residuals_on_f2,
                                     num_of_residuals_on_f3)
        
    def forward(self, im, scales, prediction_from_scales, device):
        min_prediction_scale = min(prediction_from_scales)

        # 3 -> ch1 channels
        tmp = self.first_2d_layer(im)

        # 2d processing
        out_tensors = []
        for i in range(len(scales)):
            sc = scales[i]
            extract = True if i >= min_prediction_scale else False
            if i == 0:
                tmp1, tmp2 = self.module_2d_1(tmp, sc[1:], extract, device)
            elif i == 1:
                tmp1, tmp2 = self.module_2d_2(tmp, sc[1:], extract, device)
            elif i == 2:
                tmp1, tmp2 = self.module_2d_3(tmp, sc[1:], extract, device)
            elif i == 3:
                tmp1, tmp2 = self.module_2d_4(tmp, sc[1:], extract, device)
            else:
                tmp1, tmp2 = self.module_2d_1(tmp, sc[1:], extract, device)
                
            out_tensors.append(tmp2)

        return out_tensors

# f4: patch comparison module
def patch_comparison_volume(imL, imR, imL_d, imR_d, max_disp, is_training, device, ref = 'left'):
    if ref == 'left':
        b = imL_d.size()[0]
        ch = imL_d.size()[1]
        h = imL_d.size()[2]
        w = imL_d.size()[3]

        imL_d_abs = torch.abs(imL_d)
        imR_d_abs = torch.abs(imR_d)

        imL = torch.nn.functional.interpolate(
            imL, (h,w), mode='bilinear', align_corners=True)
        imR = torch.nn.functional.interpolate(
            imR, (h,w), mode='bilinear', align_corners=True)

        vol = torch.zeros([b, ch+3, max_disp+1, h, w], dtype=torch.float32,
                          device=device, requires_grad=False)

        tmp = (imL_d_abs + imR_d_abs)/2 * torch.exp(-torch.abs(imL_d - imR_d))
        vol[:, :3, 0, :, :] = imL
        vol[:, 3:, 0, :, :] = tmp

        for i in range(1, max_disp + 1):
            vol[:, :3, i, :, i:] = imL[:,:,:,i:]

            tmp1 = imL_d[:,:,:,i:]
            tmp2 = imR_d[:,:,:,:-i]
            tmp1_abs = imL_d_abs[:,:,:,i:]
            tmp2_abs = imR_d_abs[:,:,:,:-i]
            tmp = (tmp1_abs + tmp2_abs)/2 * torch.exp(-torch.abs(tmp1 - tmp2))
            vol[:, 3:, i, :, i:] = tmp

        # assure tensor is contiguous
        if not vol.is_contiguous():
            vol = vol.contiguous()
            
    elif ref == 'right':
        b = imL_d.size()[0]
        ch = imL_d.size()[1]
        h = imL_d.size()[2]
        w = imL_d.size()[3]

        imL_d_abs = torch.abs(imL_d)
        imR_d_abs = torch.abs(imR_d)

        imL = torch.nn.functional.interpolate(
            imL, (h,w), mode='bilinear', align_corners=True)
        imR = torch.nn.functional.interpolate(
            imR, (h,w), mode='bilinear', align_corners=True)

        vol = torch.zeros([b, ch+3, max_disp+1, h, w], dtype=torch.float32,
                          device=device, requires_grad=False)

        tmp = (imL_d_abs + imR_d_abs)/2 * torch.exp(-torch.abs(imL_d - imR_d))
        vol[:, :3, 0, :, :] = imR
        vol[:, 3:, 0, :, :] = tmp

        for i in range(1, max_disp + 1):
            vol[:, :3, i, :, :-i] = imR[:,:,:,:-i]

            tmp1 = imL_d[:,:,:,i:]
            tmp2 = imR_d[:,:,:,:-i]
            tmp1_abs = imL_d_abs[:,:,:,i:]
            tmp2_abs = imR_d_abs[:,:,:,:-i]
            tmp = (tmp1_abs + tmp2_abs)/2 * torch.exp(-torch.abs(tmp1 - tmp2))
            vol[:, 3:, i, :, :-i] = tmp

        # assure tensor is contiguous
        if not vol.is_contiguous():
            vol = vol.contiguous()
        

    return vol

# helper module
class residual_3d_module(torch.nn.Module):
    # bn + relu + conv3d strategy
    def __init__(self, ch):
        super().__init__()
        self.seq1 = torch.nn.Sequential(
            torch.nn.BatchNorm3d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch, ch, 3, 1, 1, 1, bias=False)
        )

        self.seq2 = torch.nn.Sequential(
            torch.nn.BatchNorm3d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch, ch, 3, 1, 1, 1, bias=False))

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)
        out += x
        return out


# f5: for out
class for_out_3d_mod(torch.nn.Module):
    def __init__(self, ch, num_of_residuals):
        super().__init__()

        self.first_layer = torch.nn.Sequential(
            torch.nn.BatchNorm3d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch, ch, 3, 1, 1, 1, bias=False)
        )

        layers = []
        for i in range(num_of_residuals):
            layers.append(residual_3d_module(ch))
        self.seq = torch.nn.Sequential(*layers)

        self.last_layer = torch.nn.Sequential(
            torch.nn.BatchNorm3d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch, 1, 3, 1, 1, 1, bias=False)
        )

    def forward(self, x):
        out = self.first_layer(x)
        out = self.seq(out)
        out = self.last_layer(out)
        return out

class for_out_3d(torch.nn.Module):
    def __init__(self, ch, num_of_residuals):
        super().__init__()
        self.for_out_3d_0 = for_out_3d_mod(ch, num_of_residuals)
        self.for_out_3d_1 = for_out_3d_mod(ch, num_of_residuals)        
        self.for_out_3d_2 = for_out_3d_mod(ch, num_of_residuals)        
        self.for_out_3d_3 = for_out_3d_mod(ch, num_of_residuals)        
        
    def forward(self, x, i):
        if i == 0:
            out = self.for_out_3d_0(x)
        elif i == 1:
            out = self.for_out_3d_1(x)
        elif i == 2:
            out = self.for_out_3d_2(x)
        elif i == 3:
            out = self.for_out_3d_3(x)
            
        return out


# f6: 3d upsample
def upsampling_3d(x, new_dimension):
    return torch.nn.functional.interpolate(
        x, new_dimension, mode='trilinear',
        align_corners=True)


# f7: process before merging
class before_merging(torch.nn.Module):
    def __init__(self, ch, num_of_residuals):
        super().__init__()

        self.first_layer = torch.nn.Sequential(
            torch.nn.BatchNorm3d(ch + 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch+3, ch, 3, 1, 1, 1, bias=False)
        )

        layers = []
        for i in range(num_of_residuals):
            layers.append(residual_3d_module(ch))
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.seq(out)
        return out


# f8: merging
class merging_3d(torch.nn.Module):
    def __init__(self, ch, num_of_residuals):
        super().__init__()

        self.first_layer = torch.nn.Sequential(
            torch.nn.BatchNorm3d(2*ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(2*ch, 2*ch, 3, 1, 1, 1, bias=False),
            torch.nn.BatchNorm3d(2*ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(2*ch, ch, 3, 1, 1, 1, bias=False)
            
        )

        layers = []
        for i in range(num_of_residuals):
            layers.append(residual_3d_module(ch))
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x1, x2):
        if isinstance(x1, torch.Tensor):
            out = torch.cat((x1, x2), 1)
            out = self.first_layer(out)
            out = self.seq(out)
        else:
            out = x2
        return out


# wrapper module
class module_3d(torch.nn.Module):
    def __init__(self, ch, num_of_residuals_f7, num_of_residuals_f8):
        super().__init__()

        self.before_merging_0 = before_merging(ch, num_of_residuals_f7)
        self.before_merging_1 = before_merging(ch, num_of_residuals_f7)
        self.before_merging_2 = before_merging(ch, num_of_residuals_f7)
        self.before_merging_3 = before_merging(ch, num_of_residuals_f7)        
        
        self.merging_3d_0 = merging_3d(ch, num_of_residuals_f8)
        self.merging_3d_1 = merging_3d(ch, num_of_residuals_f8)
        self.merging_3d_2 = merging_3d(ch, num_of_residuals_f8)
        self.merging_3d_3 = merging_3d(ch, num_of_residuals_f8)

    def forward(self, x1, x2, i):
        # processing of x1
        if isinstance(x1, torch.Tensor):
            tmp1 = upsampling_3d(x1, x2.shape[2:])
        else:
            tmp1 = x1
        # tmp1 = self.before_merging(tmp1)

        if i == 0:
            tmp2 = self.before_merging_0(x2)
            tmp = self.merging_3d_0(tmp1, tmp2)
        elif i == 1:
            tmp2 = self.before_merging_1(x2)
            tmp = self.merging_3d_1(tmp1, tmp2)
        elif i == 2:
            tmp2 = self.before_merging_2(x2)
            tmp = self.merging_3d_2(tmp1, tmp2)
        elif i == 3:
            tmp2 = self.before_merging_3(x2)
            tmp = self.merging_3d_3(tmp1, tmp2)

        return tmp


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
