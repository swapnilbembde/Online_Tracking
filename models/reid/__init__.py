from __future__ import absolute_import

from .cpm import appearance_body_map

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from utils import bbox as bbox_utils
from utils.log import logger
from models import net_utils

import os.path as osp
from models.reid.cpm import Appearance_Body
import json


__factory = {
    'body_map': appearance_body_map,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)

def load_reid_model():
    target_epoch = 200
    exp_dir = '/home/SharedData/swapnil/Code/data/'
    args = json.load(open(osp.join(exp_dir, "args.json"), "r"))
    model = Appearance_Body(dilation=args['dilation'], use_relu=args['use_relu'], initialize=False).cuda()
    model.inp_size = (80, 160)
    weight_file = osp.join(exp_dir, 'epoch_{}.pth.tar'.format(target_epoch))
    model.load(load_checkpoint(weight_file))
    model.eval()
    return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def extract_reid_features(reid_model, image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)
    patches = np.asarray([im_preprocess(cv2.resize(p, reid_model.inp_size)) for p in patches], dtype=np.float32)

    gpu = net_utils.get_device(reid_model)
    with torch.no_grad():
        im_var = Variable(torch.from_numpy(patches))
        if gpu is not None:
            im_var = im_var.cuda(gpu)
            features = reid_model(im_var).data

    return features

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Load checkpoint from '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
