# -*- coding: utf-8 -*-
# @Time    : 19-5-5 15:55
# @Author  : yanqi
# @File    : test_denoising.py
# @IDE: PyCharm
"""
test denoising model
"""
import os
import os.path as ops
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import sys
sys.path.append('../')
from attentive_gan_model import denoising_net
from config import global_config
from IPython.core.debugger import Tracer

CFG = global_config.cfg


def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='E:/datasets/denoising/processing_data/blur_cliped_test', help='The input image path')
    parser.add_argument('--save_path', type=str, default='E:/datasets/denoising/results/attentive-gan/5.6_10000_post', help='The input image path')
    parser.add_argument('--weights_path', type=str, default='../model/denoising_gan/denoising_gan_2019-05-06-13-02-45.ckpt-10000', help='The model weights path')
    parser.add_argument('--label_path', type=str, default=None, help='The label image path')

    return parser.parse_args()


def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_model(image_path, save_path, weights_path, label_path=None):
    """
    :param image_path:
    :param weights_path:
    :param label_path:
    :return:
    """
    #assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                  name='input_tensor'
                                  )

    phase = tf.constant('test', tf.string)
    net = denoising_net.DenoisingNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='denoising_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=False)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    if not ops.exists(save_path):
        os.makedirs(save_path)

    # test images list
    images = os.listdir(image_path)

    # add filter as post process
    kernel_sharpen = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]]) / 8.0


    for i, img in enumerate(images):
        image_dir = ops.join(image_path, img)
        image = cv2.imread(image_dir, cv2.IMREAD_COLOR)
        # if image is GRAY image
        # image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        image_vis = image

        # if image is GRAY image
        #image_vis = image[:, :, np.newaxis]
        image = np.divide(np.array(image_vis, np.float32), 127.5) - 1.0

        label_image_vis = None
        if label_path is not None:
            labels = os.listdir(label_path)
            for label in labels:
                label_dir = ops.join(label_path, label)
                label_image = cv2.imread(label_dir, cv2.IMREAD_COLOR)
                # label_image = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
                # label_image = label_image[:, :, np.newaxis] #(512, 512, 1)
                label_image_vis = cv2.resize(
                    label_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            output_image = np.array(output_image, np.uint8)

            if label_path is not None:
                label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                psnr = compare_psnr(label_image_vis_gray, output_image_gray)
                ssim = compare_ssim(label_image_vis_gray, output_image_gray)

                print('SSIM: {:.5f}'.format(ssim))
                print('PSNR: {:.5f}'.format(psnr))

            # merge src image, denoising image and post process image
            out_image = np.zeros((CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH * 3, 3), dtype=np.uint8)

            # if image is GRAY image
            #out_image[:, :CFG.TEST.IMG_WIDTH] = image_vis[:, :, 0]
            out_image[:, :CFG.TEST.IMG_WIDTH] = image_vis

            # Tracer()()
            output_image_resize = cv2.resize(output_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            out_image[:, CFG.TEST.IMG_WIDTH:CFG.TEST.IMG_WIDTH * 2] = output_image_resize

            output_sharpen = cv2.filter2D(output_image, -1, kernel_sharpen)
            output_sharpen_resize = cv2.resize(output_sharpen, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            out_image[:, CFG.TEST.IMG_WIDTH * 2:] = output_sharpen_resize

            # save results
            # cv2.imwrite(save_path + '/' + img[:-4] + '_src_img.jpg', image_vis)
            # cv2.imwrite(save_path + '/' + img[:-4] + '_denoising_img.jpg', output_image)
            # cv2.imwrite(save_path + '/' + img[:-4] + '_denoising_post_img.jpg', output_sharpen)
            cv2.imwrite(save_path + '/' + img[:-4] + '_out.jpg', out_image)

            print('**************{} have tested over!**************'.format(img))

    print('_' * 50)
    print('All test images have tested successfully!')

if __name__ == '__main__':
    # init args
    args = init_args()

    # test model
    test_model(args.image_path, args.save_path, args.weights_path, args.label_path)