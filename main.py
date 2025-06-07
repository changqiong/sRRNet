from __future__ import division
import argparse
import json
import os
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
import tfmodel
from data_loader import Dataloader as inputdata

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--parameter', type=str, default=None)
args = parser.parse_args()

with open(args.parameter) as f:
    j = json.load(f)

Params = tf.contrib.framework.nest.map_structure(lambda *v: v[0], j)

param_keys = [
    'left_path', 'disp_path', 'gt_path', 'gt_path_noc',
    'val_left_path', 'val_disp_path', 'val_gt_path', 'val_gt_path_noc',
    'down_sample_ratio', 'epochs']

params = type('parameters', (), {k: j[k] for k in param_keys})()

model_save = "./model/model.ckpt"
output = "result/"
if not os.path.exists(output):
    os.mkdir(output)

def create_placeholders(channels):
    x = tf.placeholder(tf.float32, [1, None, None, channels], name="x_p")
    y = tf.placeholder(tf.float32, [1, None, None, 1], name="y_p")
    y_noc = tf.placeholder(tf.float32, [1, None, None, 1], name="y_noc_p")
    input_height = tf.Variable(0, name="input_height", dtype=tf.int32)
    input_width = tf.Variable(0, name="input_width", dtype=tf.int32)
    is_training = tf.Variable(True, name="is_training", dtype=tf.bool)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return x, y, y_noc, input_height, input_width, is_training, keep_prob

def run_prediction(sess, pred, feed):
    return sess.run(pred, feed_dict=feed)

def run_validation(sess, pred, error_fn, dataloader, feed_fn):
    dataloader.init_sample_index(0)
    acc1, acc2 = np.empty([0]), np.empty([0])
    while dataloader.get_sample_index() < 200:
        d, gt, noc, idx = dataloader.load_validation_sample()
        err = sess.run(error_fn, feed_fn(d, gt, noc))
        (acc1 if idx <= 160 else acc2)[:] = np.append((acc1 if idx <= 160 else acc2), err)
    m1, m2 = np.mean(acc1), np.mean(acc2)
    with open('correct_rate.txt', 'a') as f:
        f.write('epoch:{} ahead160 {:.4f} last40 {:.4f}\n'.format(dataloader.epoch, m1*100, m2*100))
    print("Epoch: {} Validation error: {:.4f}".format(dataloader.epoch, m1))
    return m1

with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name="g_step")
    lr = tf.train.exponential_decay(1e-4, global_step, 1600, 0.98, staircase=True)

    if args.mode == 'train':
        train_loader = inputdata(params)
        val_loader = inputdata(params)
        h, w, c = train_loader.get_training_data_size()

        with tf.Session() as sess:
            x, y, y_noc, ih, iw, training, kp = create_placeholders(c)
            max_disp = tf.Variable(0, name="max_disp", dtype=tf.int32)
            pred = tf.to_float(tfmodel.main_net(x, ih, iw, training, False, kp))
            out_1 = pred[:, :, :, 0:1]

            weights = tf.cast(tf.greater(y, 0), tf.float32)
            loss1 = 1 - tf.reduce_mean(tf.image.ssim(y * weights, out_1 * weights, max_val=128.0))
            loss2 = tf.losses.absolute_difference(y, out_1, weights=weights)
            loss = 0.5 * (loss1 + loss2)
            opt = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            error = tfmodel.gt_compare(out_1, y)

            saver = tf.train.Saver(max_to_keep=0)
            sess.run(tf.global_variables_initializer())
            if args.model:
                saver.restore(sess, args.model)
                print("Model restored")

            acc_err, epoch_now = np.empty([0]), train_loader.epoch
            while train_loader.epoch < params.epochs:
                if epoch_now < train_loader.epoch:
                    print("Epoch: {} Training error: {:.4f}".format(train_loader.epoch, np.mean(acc_err)))
                    acc_err, epoch_now = np.empty([0]), train_loader.epoch

                d, gt, noc, idx = train_loader.load_training_sample()
                fd = {x: d, y: gt, y_noc: noc, iw: w, ih: h, training: True, max_disp: train_loader.max_disp, kp: 0.8}
                _, l, e = sess.run([opt, loss, error], feed_dict=fd)
                acc_err = np.append(acc_err, e)

                if (idx == 1 and train_loader.epoch > 399) or (idx == 1 and train_loader.epoch % 10 == 0):
                    v_h, v_w, _ = val_loader.get_data_size()
                    run_validation(sess, pred, error, val_loader,
                                   lambda d, g, n: {x: d, y: g, y_noc: n, iw: v_w, ih: v_h,
                                                    training: False, max_disp: val_loader.max_disp, kp: 1})
                    saver.save(sess, model_save, global_step=global_step)

    else:
        with tf.Session(config=config) as sess:
            dataloader = inputdata(params)
            h, w, c = dataloader.get_data_size()
            x, _, _, ih, iw, training, kp = create_placeholders(c)

            pred = tf.to_float(tfmodel.main_net(x, ih, iw, training, False, kp))
            saver = tf.train.Saver()
            saver.restore(sess, args.model)
            print("Model restored")

            t_total, t_infer = time.time(), 0
            while dataloader.get_sample_index() < dataloader.get_sample_size():
                t1 = time.time()
                d, idx, oh, ow, name = dataloader.load_test_sample()
                print(time.time() - t1)
                t2 = time.time()
                out = sess.run(pred, feed_dict={x: d, iw: w, ih: h, training: False, kp: 1})
                t_infer += time.time() - t2 if idx != 1 else 0

                disp = np.uint16(out[0, out.shape[1]-oh:, out.shape[2]-ow:, 0] * 256)
                cv.imwrite(os.path.join(output, name), disp)
                print("Res saved at: ./test/{}".format(name))

            print("total time:", time.time() - t_total)
