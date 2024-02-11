﻿import sys

sys.path.insert(0, "..")
import os
import random
import glob
import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from skimage import color
from torch.autograd import Variable
from utils.flowlib import read_flow
from utils.util_distortion import CenterPad,CenterPad_vec
import torchvision.transforms as transform_lib
import lib.functional as F

cv2.setNumThreads(0)


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return color.rgb2lab(inputs)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        inputs[0:1, :, :] = F.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = F.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        outputs = F.to_mytensor(inputs)  # permute channel and transform to tensor
        return outputs


class RandomErasing(object):
    def __init__(self, probability=0.6, sl=0.05, sh=0.6):
        self.probability = probability
        self.sl = sl
        self.sh = sh

    def __call__(self, img):
        img = np.array(img)
        if random.uniform(0, 1) > self.probability:
            return Image.fromarray(img)

        area = img.shape[0] * img.shape[1]
        h0 = img.shape[0]
        w0 = img.shape[1]
        channel = img.shape[2]

        h = int(round(random.uniform(self.sl, self.sh) * h0))
        w = int(round(random.uniform(self.sl, self.sh) * w0))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1 : x1 + h, y1 : y1 + w, :] = np.random.rand(h, w, channel) * 255
            return Image.fromarray(img)

        return Image.fromarray(img)


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy


def parse_images(data_root,flow_root,mask_root,gap):
    image_pairs = []
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        path = os.path.join(data_root, subdir)
        flow_path= os.path.join(flow_root, subdir)
        mask_path= os.path.join(mask_root, subdir)
        if not os.path.isdir(path):
            continue

        #parse_file = os.path.join(path, "pairs_output_new.txt")
        imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
        flows=sorted(glob.glob(os.path.join(flow_path, '*.flo')))
        masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))
        reference = imgs[0]
        item=[]
        for i in range(len(imgs)-1):   
            item.append([
                imgs[i],
                imgs[i+1],
                flows[i],
                masks[i],
                reference,]
            )
            if (i+1) % gap == 0:
                reference = imgs[i+1]
                image_pairs.append(item)       # a list with length gap
                item = []
                    #if i > 50:
                    #    break

        #else:
        #    raise (RuntimeError("Error when parsing pair_output_count.txt in subfolders of: " + path + "\n"))

    return image_pairs


class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        flow_root,
        mask_root,
        epoch,
        image_size,
        image_transform=None,
        reference_transform = None,
        extra_reference_transform =None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.img_h = image_size[1]
        self.img_w = image_size[0]
        self.data_root = data_root
        self.flow_root=flow_root
        self.mask_root = mask_root
        self.image_transform = image_transform
        self.reference_transform = reference_transform
        self.extra_reference_transform = extra_reference_transform
        self.CenterPad_vec = CenterPad_vec([self.img_w,self.img_h])
        self.CenterPad = CenterPad([self.img_w,self.img_h])
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images(self.data_root,self.flow_root,self.mask_root,10)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        item = self.image_pairs[index]
        #print("item:",item[0][0])
        outputs=[]
        for [
            image1_name,
            image2_name,
            flow_name,
            mask_name,
            reference_gt1,
        ] in item:
        
            I1 = Image.open(image1_name)
            I2 = Image.open(image2_name)

            I_reference_output = Image.open(reference_gt1)
            try:
                flow = read_flow(flow_name)  # numpy
            except:
                print("error when processing:",flow_name)
            mask = Image.open(mask_name)
            # binary mask
            #mask = np.array(mask)
            #mask[mask < 240] = 0
            #mask[mask >= 240] = 1

            # transform
            #print("shape:I1 ",I1.size,"shape:I2 ",I2.size)
            #I1 = np.array(I1)
            #I2 = np.array(I2)
            #print("centerpad!")
            I1 = self.image_transform(self.CenterPad(I1))
            I2 = self.image_transform(self.CenterPad(I2))
            #print("shape:I1_trans ",I1.shape,"shape:I2_trans ",I2.shape)
            #print("shape:I_ref ",I_reference_video.size)
            #------------------reference argumentation-------------#
            if  self.reference_transform :
                I_reference_output = self.reference_transform(I_reference_output)
            if self.extra_reference_transform:
                if np.random.random() < 0.1 :
                      I_reference_output = self.extra_reference_transform(I_reference_output)
            I_reference_output = self.image_transform(self.CenterPad(I_reference_output))			  #centerCrop centerPad 是干嘛的            这里输出的reference是不是有问题。   
            #print("centerpad_vec!")
            #print("flow:",flow.shape,"mask:",mask.shape,"\n image1:",I1.shape)
            flow = self.ToTensor(self.CenterCrop(self.CenterPad_vec(flow)))
            mask = self.ToTensor((self.CenterPad(mask)))
           # print("mean:",mask.mean())
            mask[mask < 125] = 0
            mask[mask >= 125] = 1
            outputs.append( [
                I1,
                I2,
                flow,
                mask,
                I_reference_output,
            ])
        return outputs

    def __len__(self):
        return len(self.image_pairs)


def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    l_norm, ab_norm = 1.0, 1.0
    l_mean, ab_mean = 50.0, 0
    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")
