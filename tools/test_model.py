#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test model
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

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='E:/datasets/denoising/processing_data/blur_cliped_test/20171021124604_0001.jpg', help='The input image path')
    parser.add_argument('--weights_path', type=str, default='../model/denoising_gan/denoising_gan_2019-05-06-18-21-26.ckpt-100000', help='The model weights path')

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


def test_model(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    #assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 1],
                                  name='input_tensor'
                                  )

    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    #image_vis = image
    image_vis = image[:, :, np.newaxis]
    image = np.divide(np.array(image_vis, np.float32), 127.5) - 1.0

    phase = tf.constsant('test', tf.string)

    net = denoising_net.DenoisingNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='denoising_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=False)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        output_image, atte_maps = sess.run(
            [output, attention_maps],
            feed_dict={input_tensor: np.expand_dims(image, 0)})

        output_image = output_image[0]
        for i in range(output_image.shape[2]):
            output_image[:, :, i] = minmax_scale(output_image[:, :, i])

        output_image = np.array(output_image, np.uint8)


        # 保存并可视化结果
        save_path = '../test_results'
        if not ops.exists(save_path):
            os.makedirs(save_path)

        # add post process
        kernel_sharpen = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]]) / 8.0

        output_sharpen = cv2.filter2D(output_image, -1, kernel_sharpen)

        cv2.imwrite(save_path + '/' + 'src_img.png', image_vis)
        cv2.imwrite(save_path + '/' + 'denoising_ret.png', output_image)
        cv2.imwrite(save_path + '/' + 'denoising_ret_post.png', output_sharpen)
        print('_' * 50)
        print('test results have saved successfully!')

        #————————————————————后处理实验———————————————————————————
        """
       
        # add post processing eg. image enhance
        # 1）形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) # 定义矩形结构元素
        # *** // 针对白色区域，先腐蚀后膨胀（开运算），等效于对文字进行了闭运算
        output_closed = cv2.morphologyEx(output_image, cv2.MORPH_OPEN, kernel, iterations=3)

        # 2) 锐化
        # define kernel
        kernel_sharpen1 = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]])

        kernel_sharpen2 = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]]) / 8.0

        kernel_sharpen3 = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1,  2,  2,  2,  2,  2, -1],
            [-1,  2,  4,  4,  4,  2, -1],
            [-1,  2,  4,  8,  4,  2, -1],
            [-1,  2,  4,  4,  4,  2, -1],
            [-1,  2,  2,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1]]) / 48.0

        output_sharpen1 = cv2.filter2D(output_image, -1, kernel_sharpen1)
        output_sharpen2 = cv2.filter2D(output_image, -1, kernel_sharpen2)
        output_sharpen3 = cv2.filter2D(output_image, -1, kernel_sharpen3)

        # 3) 先锐化，再闭运算 ***//针对白色区域，先腐蚀后膨胀（开运算），等效于对文字进行了闭运算
        output_sharpen_closed1 = cv2.morphologyEx(output_sharpen1, cv2.MORPH_OPEN, kernel, iterations=3)
        output_sharpen_closed2 = cv2.morphologyEx(output_sharpen2, cv2.MORPH_OPEN, kernel, iterations=3)
        output_sharpen_closed3 = cv2.morphologyEx(output_sharpen3, cv2.MORPH_OPEN, kernel, iterations=3)

        # 4) 先锐化，再膨胀 ***//使用了Erode方法，腐蚀操作，针对白色区域，等效于对文字进行了膨胀
        output_sharpen_dilated1 = cv2.erode(output_sharpen1, kernel)
        output_sharpen_dilated2 = cv2.erode(output_sharpen2, kernel)
        output_sharpen_dilated3 = cv2.erode(output_sharpen3, kernel)

        cv2.imshow('src_image', image_vis)
        cv2.imshow('denoising_image', output_image)

        cv2.imshow('close1_image', output_closed)

        cv2.imshow('sharpen1_image', output_sharpen1)
        cv2.imshow('sharpen2_image', output_sharpen2)
        cv2.imshow('sharpen3_image', output_sharpen3)

        cv2.imshow('sharpen_close1_image', output_sharpen_closed1)
        cv2.imshow('sharpen_close2_image', output_sharpen_closed2)
        cv2.imshow('sharpen_close3_image', output_sharpen_closed3)

        cv2.imshow('sharpen_dilated1_image', output_sharpen_dilated1)
        cv2.imshow('sharpen_dilated2_image', output_sharpen_dilated2)
        cv2.imshow('sharpen_dilated3_image', output_sharpen_dilated3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        #-------------------初步结论：使用5*5的滤波器即可，目前加形态学处理似乎并没有什么作用---------------------------

        # plt.figure('src_image')
        # plt.imshow(image_vis[:, :, (0, 0, 0)])
        # plt.figure('denoising_ret')
        # plt.imshow(output_image[:, :, (0, 0, 0)])
        # plt.figure('atte_map_1')
        # plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_1.png')
        # plt.figure('atte_map_2')
        # plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_2.png')
        # plt.figure('atte_map_3')
        # plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_3.png')
        # plt.figure('atte_map_4')
        # plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_4.png')
        # plt.show()

        # plt.figure('src_image')
        # plt.imshow(image_vis[:, :, (2, 1, 0)])
        # plt.figure('derain_ret')
        # plt.imshow(output_image[:, :, (2, 1, 0)])
        # plt.figure('atte_map_1')
        # plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_1.png')
        # plt.figure('atte_map_2')
        # plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_2.png')
        # plt.figure('atte_map_3')
        # plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_3.png')
        # plt.figure('atte_map_4')
        # plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
        # plt.savefig('atte_map_4.png')
        # plt.show()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test model
    test_model(args.image_path, args.weights_path)
