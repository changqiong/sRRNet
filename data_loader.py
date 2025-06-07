# ============================================================================
# Copyright (c) 2022, Chang Qiong. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Qiong Chang
# Description: Refactored main execution file for sRRNet training and testing
# Compatible with Python 2.7 and TensorFlow 1.9.0-GPU
# ============================================================================

from __future__ import division
import tensorflow as tf
import numpy as np
import scipy
import os
import time
import random

class Dataloader(object):
    def __init__(self, params):
        self.__params = params
        self.__contents = sorted(os.listdir(self.__params.val_disp_path))
        self.__tr_contents = os.listdir(self.__params.disp_path)
        self.__training_samples = len(self.__contents)
        self.__sample_index = 0
        self.epoch = 0
        self.maxwidth = 0
        self.maxheight = 0
        self.__init_index = 0
        self.configure_input_size()
        self.__widthresize = self.maxwidth + (self.__params.down_sample_ratio - self.maxwidth % self.__params.down_sample_ratio) % self.__params.down_sample_ratio
        self.__heightresize = self.maxheight + (self.__params.down_sample_ratio - self.maxheight % self.__params.down_sample_ratio) % self.__params.down_sample_ratio
        self.max_disp = 128
        self.__val_num = 40

    def init_sample_index(self, val):
        self.__sample_index = val

    def get_sample_size(self):
        return self.__training_samples

    def get_sample_index(self):
        return self.__sample_index

    def get_data_size(self):
        return self.__heightresize, self.__widthresize, 2

    def get_training_data_size(self):
        return self.get_data_size()

    def configure_input_size(self):
        for fname in self.__contents:
            img = scipy.misc.imread(self.__params.val_gt_path + fname).astype(float)
            self.maxheight = max(self.maxheight, img.shape[0])
            self.maxwidth = max(self.maxwidth, img.shape[1])

    def fold_1(self):
        self.__training_samples //= 2
        self.__contents = self.__contents[:self.__training_samples]

    def fold_2(self):
        self.__init_index = self.__training_samples // 2
        self.__contents = self.__contents[self.__init_index:]
        self.__training_samples //= 2

    def shuffle_data(self):
        np.random.seed(int(time.time()))
        np.random.shuffle(self.__tr_contents)

    def _pad(self, img, target_h, target_w):
        pad_h = max(target_h - img.shape[0], 0)
        pad_w = max(target_w - img.shape[1], 0)
        return np.lib.pad(img, [(pad_h, 0), (pad_w, 0)], 'edge')

    def _prepare_sample(self, img, disp, gt=None, gt_noc=None):
        h, w = self.__heightresize, self.__widthresize
        mh, mw = h // 2, w // 2

        img = self._pad(img, mh, mw)
        disp = self._pad(disp, mh, mw)
        data = np.stack([disp, img], axis=2)
        data = np.reshape(data, [1, data.shape[0], data.shape[1], 2])

        if gt is not None and gt_noc is not None:
            gt = self._pad(gt, h, w)
            gt_noc = self._pad(gt_noc, h, w)
            gt = np.reshape(gt, [1, gt.shape[0], gt.shape[1], 1])
            gt_noc = np.reshape(gt_noc, [1, gt_noc.shape[0], gt_noc.shape[1], 1])

        return data, gt, gt_noc

    def load_training_sample(self):
        if self.__sample_index >= self.__training_samples - self.__val_num:
            self.__sample_index = 0
            self.epoch += 1
            self.shuffle_data()

        fname = self.__tr_contents[self.__sample_index]
        img = scipy.misc.imread(self.__params.left_path + fname).astype(float)
        disp = scipy.misc.imread(self.__params.disp_path + fname).astype(float) / 256
        gt = scipy.misc.imread(self.__params.gt_path + fname).astype(float) / 256
        gt_noc = scipy.misc.imread(self.__params.gt_path_noc + fname).astype(float) / 256

        data, gt, gt_noc = self._prepare_sample(img, disp, gt, gt_noc)
        self.__sample_index += 1
        return data, gt, gt_noc, self.__sample_index

    def load_validation_sample(self):
        fname = self.__contents[self.__sample_index]
        img = scipy.misc.imread(self.__params.val_left_path + fname).astype(float)
        disp = scipy.misc.imread(self.__params.val_disp_path + fname).astype(float) / 256
        gt = scipy.misc.imread(self.__params.val_gt_path + fname).astype(float) / 256
        gt_noc = scipy.misc.imread(self.__params.val_gt_path_noc + fname).astype(float) / 256

        data, gt, gt_noc = self._prepare_sample(img, disp, gt, gt_noc)
        self.__sample_index += 1
        return data, gt, gt_noc, self.__sample_index

    def load_verify_sample(self):
        if self.__sample_index >= self.__training_samples:
            self.__sample_index = self.__training_samples - 40

        fname = self.__contents[self.__sample_index]
        img = scipy.misc.imread(self.__params.left_path + fname).astype(float)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        disp = scipy.misc.imread(self.__params.disp_path + fname).astype(float) / 256
        gt = scipy.misc.imread(self.__params.gt_path + fname).astype(float) / 256
        gt_noc = scipy.misc.imread(self.__params.gt_path_noc + fname).astype(float) / 256

        data, gt, gt_noc = self._prepare_sample(img, disp, gt, gt_noc)
        name = fname
        self.__sample_index += 1
        return data, gt, gt_noc, self.__sample_index, img.shape[0], img.shape[1], name

    def load_test_sample(self):
        if self.__sample_index >= self.__training_samples:
            self.__sample_index = 0

        fname = self.__contents[self.__sample_index]
        print(self.__params.val_left_path + fname)
        img = scipy.misc.imread(self.__params.val_left_path + fname).astype(float)
        disp = scipy.misc.imread(self.__params.val_disp_path + fname).astype(float) / 256
        test_left = scipy.misc.imread(self.__params.left_path + fname).astype(float)

        height, width = test_left.shape
        img = self._pad(img, self.__heightresize // 2, self.__widthresize // 2)
        disp = self._pad(disp, self.__heightresize // 2, self.__widthresize // 2)

        data = np.stack([disp, img], axis=2)
        data = np.reshape(data, [1, data.shape[0], data.shape[1], 2])
        name = fname
        self.__sample_index += 1
        return data, self.__sample_index, height, width, name
