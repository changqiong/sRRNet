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

import tensorflow as tf
import utils

_FLOAT = tf.float32
_SAME = 'SAME'

def get_error_map(x, gt):
    res = tf.abs(x * tf.cast(tf.greater(gt, 0), _FLOAT) - gt)
    return tf.cast(tf.greater(res, 3.0), _FLOAT)

def gt_compare(x, gt):
    mask = tf.cast(tf.greater(gt, 0), _FLOAT)
    valid = tf.cast(tf.count_nonzero(gt), _FLOAT)
    err = tf.reduce_sum(tf.cast(tf.greater(tf.abs(x * mask - gt), 3.0), _FLOAT))
    return err / valid

def res_block(inputs, kernel, features, scope, dropout, bn, reuse, is_training):
    conv = utils.conv2d(inputs, features, kernel, scope + '_block_conv1',
                        stride=[1, 1], padding=_SAME, use_xavier=True,
                        stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                        bn=bn, bn_decay=None, is_training=is_training,
                        dropout=dropout, reuse=reuse)
    return tf.concat([inputs, conv], axis=3)

def main_net(inputs, height, width, is_training, reuse, keep_prob):
    bn, bn2 = False, False

    conv1 = utils.conv2d_depth(inputs, 32, [5, 5], 'First_Block_conv1',
                               stride=[1, 1], padding=_SAME, rate=[1],
                               use_xavier=True, stddev=1e-3, weight_decay=0.0,
                               activation_fn=tf.nn.relu, bn=bn2, bn_decay=None,
                               is_training=is_training, dropout=None, reuse=reuse)

    conv1_block = utils.conv2d(conv1, 16, [5, 5], 'First_Block_conv1_down',
                               stride=[2, 2], padding=_SAME, use_xavier=True,
                               stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                               bn=bn, bn_decay=None, is_training=is_training,
                               dropout=keep_prob, reuse=reuse)

    conv2_block = utils.conv2d(conv1_block, 32, [5, 5], 'Second_Block_out_conv2',
                               stride=[2, 2], padding=_SAME, use_xavier=True,
                               stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                               bn=bn, bn_decay=None, is_training=is_training,
                               dropout=keep_prob, reuse=reuse)

    conv2_block = res_block(conv2_block, [5, 5], 32, 'First_Block_block2', keep_prob, bn, reuse, is_training)

    up2 = utils.conv2d_transpose(conv2_block, 16, [5, 5], 'Third_Block_inverted_conv2_up',
                                 stride=[2, 2], padding=_SAME, use_xavier=True,
                                 stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                                 bn=bn2, bn_decay=None, is_training=is_training,
                                 dropout=1, height=height // 8, width=width // 8,
                                 reuse=reuse)

    res5 = res_block(tf.concat([up2, conv1_block], axis=3), [5, 5], 16,
                     'First_Block_block5', keep_prob, bn, reuse, is_training)

    up1 = utils.conv2d_transpose(res5, 16, [5, 5], 'Third_Block_inverted_conv3_up',
                                 stride=[2, 2], padding=_SAME, use_xavier=True,
                                 stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                                 bn=bn2, bn_decay=None, is_training=is_training,
                                 dropout=1, height=height // 4, width=width // 4,
                                 reuse=reuse)

    output = tf.concat([up1, conv1, inputs[:, :, :, 0:1]], axis=3)

    return utils.conv2d_transpose(output, 1, [5, 5], 'output_layer',
                                  stride=[2, 2], padding=_SAME, use_xavier=True,
                                  stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                                  bn=False, bn_decay=None, is_training=is_training,
                                  dropout=1, height=height // 2, width=width // 2,
                                  reuse=reuse)
