# WearMask3D
# Copyright 2021 Hanjo Kim and Minsoo Kim. All rights reserved.
# http://github.com/jhh37/wearmask3d
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: rlakswh@gmail.com      (Hanjo Kim)
#         devkim1102@gmail.com   (Minsoo Kim)

import json
import random

import dlib
import pygame
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageEnhance

import models.mobilenet_v1 as mobilenet_v1
from obj_loader import *


def list_split(origin_list, n):
    k,m = divmod(len(origin_list), n)
    return (origin_list[i*k + min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_total_file_list(src_root, dst_root):
    file_list = []
    dst_file_list = []

    for root, dirs, files in os.walk(src_root, topdown=True):
        new_root = root.replace(src_root, dst_root)

        if not os.path.exists(new_root):
            os.makedirs(new_root)

        if files:
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_list.append(file_path)
                dst_file_list.append(file_path.replace(src_root, dst_root))

    return file_list, dst_file_list


def mask_transformation(br, lp):
    sum_of_mask_weights, w_acc, maskIdx = 0, 0, 0

    # get configurations
    with open('config.json') as json_file:
        configs = json.load(json_file)

    br_th = configs["brightnessThreshold"]
    lp_th = configs["laplacianVarianceThreshold"]
    masks = configs["masks"]

    num_masks = len(masks)

    for i in range(num_masks):
        sum_of_mask_weights += masks[i]['weight']

    random_value = random.randrange(0, sum_of_mask_weights)

    for i in range(num_masks):
        w_acc += masks[i]['weight']
        if (random_value < w_acc):
            maskIdx = i
            break

    mask_file_name = masks[maskIdx]['name']
    min_mask_size = masks[maskIdx]['minSize']
    mask_shape = masks[maskIdx]['shape']
    mask_surf_type = masks[maskIdx]['surface']

    enhancer = ImageEnhance.Brightness(Image.open(mask_file_name))
    enhanced_im = enhancer.enhance(0.7 + 0.3 * min(br_th, br) / br_th)

    if (lp < lp_th):
        w, h = enhanced_im.size
        enhanced_im = enhanced_im.resize((int(w * max(min_mask_size, lp / lp_th)), int(h * max(min_mask_size, lp / lp_th))))

    # enhanced_im.save("mask08.png")
    data = enhanced_im.tobytes()
    size = enhanced_im.size
    mode = enhanced_im.mode

    mask_surf = pygame.image.fromstring(data, size, mode)


    return mask_surf, mask_shape, mask_surf_type


def get_models():
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)

    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)

    cudnn.benchmark = True

    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    return model, face_detector, face_regressor