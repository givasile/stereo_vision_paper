import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# random.seed(1)

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def transformations():
    t_list = [transforms.ToTensor(),
              transforms.Normalize(**__imagenet_stats)]
    return transforms.Compose(t_list)


class dataset(Dataset):
    def __init__(self, split_dataset_inst, which, form, limit_maxD, ref_img='left'):
        """
        Input:
        - split_dataset_inst: instance of merged_dataset class
        - which: 'train', 'val' or 'test' set
        - form in ['crop', 'full_im', 'downsample']
        - limit_maxD: True/False

        Output:
        - ready for feeding inputs
        """
        assert which in ['train', 'val', 'test']
        assert form in ['crop', 'full_im']
        assert limit_maxD in [True, False]
        self._split_dataset = split_dataset_inst
        self.which = which
        self.form = form
        self.limit_maxD = limit_maxD
        self.ref_img = ref_img

    def __getitem__(self, index):
        # load subject
        sub = self._split_dataset.load(index, self.which)
        # print('Loading subject with index ' + str(sub.index) +
        #       ', from dataset: ' + sub.dataset)

        if self.ref_img == 'left':
            if self.form == 'crop':
                if sub.dataset in ['kitti_2012', 'kitti_2015']:
                    # hyperparams
                    th, tw = 256, 768

                    # crop
                    h, w = sub.imL_rgb.height, sub.imL_rgb.width
                    # random.seed(1)
                    rand_h = random.randint(0, h - th)
                    rand_w = random.randint(0, w - tw)
                    print(rand_h)
                    imL = sub.imL_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))
                    imR = sub.imR_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))

                    if self.which in ['train', 'val']:
                        dispL = sub.gtL[rand_h:rand_h+th, rand_w:rand_w+tw]
                        maskL = sub.maskL[rand_h:rand_h+th, rand_w:rand_w+tw]
                        mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                        maskL = np.logical_and(mask0, maskL)
                        dispL = torch.from_numpy(dispL.copy()).float()
                        maskL = torch.from_numpy(maskL.astype(np.uint8).copy())
                    else:
                        maskL = None
                        dispL = None
                else:
                    # hyperparams
                    th, tw = 256, 512

                    # crop
                    h, w = sub.imL_rgb.height, sub.imL_rgb.width
                    # random.seed(1)
                    rand_h = random.randint(0, h - th)
                    rand_w = random.randint(0, w - tw)
                    print(rand_h)
                    imL = sub.imL_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))
                    imR = sub.imR_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))

                    dispL = sub.gtL[rand_h:rand_h+th, rand_w:rand_w+tw]
                    maskL = sub.maskL[rand_h:rand_h+th, rand_w:rand_w+tw]
                    mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                    maskL = np.logical_and(mask0, maskL)
                    dispL = torch.from_numpy(dispL.copy()).float()
                    maskL = torch.from_numpy(maskL.astype(np.uint8).copy())

                # preproccess: Normalize with imagenet statistics
                proccess = transformations()
                imL = proccess(imL)
                imR = proccess(imR)
                return imL, imR, dispL, maskL
            elif self.form == 'full_im':
                # hyperparams
                if sub.dataset in ['kitti_2012', 'kitti_2015']:
                    th, tw = 368, 1232
                else:
                    th, tw = 544, 960

                # crop at close to full resolution
                h, w = sub.imL_rgb.height, sub.imL_rgb.width

                imL = sub.imL_rgb.crop((w-tw, h-th, w, h))
                imR = sub.imR_rgb.crop((w-tw, h-th, w, h))

                # kitti dataset case
                if sub.dataset in ['kitti_2012', 'kitti_2015']:
                    if self.which in ['train', 'val']:
                        diffh = h - th
                        diffw = w - tw

                        dispL = Image.fromarray(sub.gtL).crop((w-tw, h-th, w, h))
                        dispL = np.array(dispL)
                        maskL = sub.maskL

                        if diffh < 0:
                            tmph = np.zeros((-diffh, sub.gtL.shape[1]))
                            maskL = np.vstack((tmph, maskL))
                        else:
                            maskL = maskL[diffh:, :]

                        if diffw < 0:
                            tmpw = np.zeros((maskL.shape[0], -diffw))
                            maskL = np.hstack((tmpw, maskL))
                        else:
                            maskL = maskL[:, diffw:]

                        mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                        maskL = np.logical_and(mask0, maskL)
                        dispL = torch.from_numpy(dispL.copy()).float()
                        maskL = torch.from_numpy(maskL.astype(np.uint8).copy())
                    else:
                        dispL = None
                        maskL = None
                # freiburg datasets
                else:
                    tmp = np.zeros((4, sub.gtL.shape[1]))
                    dispL = np.vstack((tmp, sub.gtL))
                    maskL = np.vstack((tmp, sub.maskL))
                    mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                    maskL = np.logical_and(mask0, maskL)
                    dispL = torch.from_numpy(dispL.copy()).float()
                    maskL = torch.from_numpy(maskL.astype(np.uint8).copy())

                # preproccess: Normalize with imagenet statistics
                proccess = transformations()
                imL = proccess(imL)
                imR = proccess(imR)
                return imL, imR, dispL, maskL
            
        # both images    
        elif self.ref_img == 'both':
            if self.form == 'crop':
                # hyperparams
                th, tw = 256, 512

                # crop
                h, w = sub.imL_rgb.height, sub.imL_rgb.width
                # random.seed(1)
                rand_h = random.randint(0, h - th)
                rand_w = random.randint(0, w - tw)
                print(rand_h)
                imL = sub.imL_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))
                imR = sub.imR_rgb.crop((rand_w, rand_h, rand_w + tw, rand_h + th))

                if sub.dataset in ['kitti_2015']:
                    if self.which in ['train', 'val']:
                        dispL = sub.gtL[rand_h:rand_h+th, rand_w:rand_w+tw]
                        maskL = sub.maskL[rand_h:rand_h+th, rand_w:rand_w+tw]
                        mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                        maskL = np.logical_and(mask0, maskL)
                        dispL = torch.from_numpy(dispL.copy()).float()
                        maskL = torch.from_numpy(maskL.astype(np.uint8).copy())

                        dispR = sub.gtR[rand_h:rand_h+th, rand_w:rand_w+tw]
                        maskR = sub.maskR[rand_h:rand_h+th, rand_w:rand_w+tw]
                        mask0 = dispR < 192 if self.limit_maxD else dispR < 100000
                        maskR = np.logical_and(mask0, maskR)
                        dispR = torch.from_numpy(dispR.copy()).float()
                        maskR = torch.from_numpy(maskR.astype(np.uint8).copy())                       
                    else:
                        maskL = None
                        dispL = None
                        maskR = None
                        dispR = None
                else:
                    dispL = sub.gtL[rand_h:rand_h+th, rand_w:rand_w+tw]
                    maskL = sub.maskL[rand_h:rand_h+th, rand_w:rand_w+tw]
                    mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                    maskL = np.logical_and(mask0, maskL)
                    dispL = torch.from_numpy(dispL.copy()).float()
                    maskL = torch.from_numpy(maskL.astype(np.uint8).copy())

                    dispR = sub.gtR[rand_h:rand_h+th, rand_w:rand_w+tw]
                    maskR = sub.maskR[rand_h:rand_h+th, rand_w:rand_w+tw]
                    mask0 = dispR < 192 if self.limit_maxD else dispR < 100000
                    maskR = np.logical_and(mask0, maskR)
                    dispR = torch.from_numpy(dispR.copy()).float()
                    maskR = torch.from_numpy(maskR.astype(np.uint8).copy())

                # preproccess: Normalize with imagenet statistics
                proccess = transformations()
                imL = proccess(imL)
                imR = proccess(imR)
                return imL, imR, dispL, maskL, dispR, maskR
            elif self.form == 'full_im':
                # hyperparams
                if sub.dataset in ['kitti_2012', 'kitti_2015']:
                    th, tw = 368, 1232
                else:
                    th, tw = 544, 960

                # crop at close to full resolution
                h, w = sub.imL_rgb.height, sub.imL_rgb.width

                imL = sub.imL_rgb.crop((w-tw, h-th, w, h))
                imR = sub.imR_rgb.crop((w-tw, h-th, w, h))

                # kitti dataset case
                if sub.dataset in ['kitti_2012', 'kitti_2015']:
                    if self.which in ['train', 'val']:
                        diffh = h - th
                        diffw = w - tw

                        dispL = Image.fromarray(sub.gtL).crop((w-tw, h-th, w, h))
                        dispL = np.array(dispL)
                        maskL = sub.maskL

                        if diffh < 0:
                            tmph = np.zeros((-diffh, sub.gtL.shape[1]))
                            maskL = np.vstack((tmph, maskL))
                        else:
                            maskL = maskL[diffh:, :]

                        if diffw < 0:
                            tmpw = np.zeros((maskL.shape[0], -diffw))
                            maskL = np.hstack((tmpw, maskL))
                        else:
                            maskL = maskL[:, diffw:]

                        mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                        maskL = np.logical_and(mask0, maskL)
                        dispL = torch.from_numpy(dispL.copy()).float()
                        maskL = torch.from_numpy(maskL.astype(np.uint8).copy())
                    else:
                        dispL = None
                        maskL = None
                # freiburg datasets
                else:
                    tmp = np.zeros((4, sub.gtL.shape[1]))
                    dispL = np.vstack((tmp, sub.gtL))
                    maskL = np.vstack((tmp, sub.maskL))
                    mask0 = dispL < 192 if self.limit_maxD else dispL < 100000
                    maskL = np.logical_and(mask0, maskL)
                    dispL = torch.from_numpy(dispL.copy()).float()
                    maskL = torch.from_numpy(maskL.astype(np.uint8).copy())

                    tmp = np.zeros((4, sub.gtR.shape[1]))
                    dispR = np.vstack((tmp, sub.gtR))
                    maskR = np.vstack((tmp, sub.maskR))
                    mask0 = dispR < 192 if self.limit_maxD else dispR < 100000
                    maskR = np.logical_and(mask0, maskR)
                    dispR = torch.from_numpy(dispR.copy()).float()
                    maskR = torch.from_numpy(maskR.astype(np.uint8).copy())

                # preproccess: Normalize with imagenet statistics
                proccess = transformations()
                imL = proccess(imL)
                imR = proccess(imR)
                return imL, imR, dispL, maskL, dispR, maskR

            

    def __len__(self):
        return len(getattr(self._split_dataset, self.which))
