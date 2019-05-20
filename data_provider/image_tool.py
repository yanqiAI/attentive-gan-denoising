# coding:utf-8
import os
import sys
import cv2
import numpy as np
import random, shutil
from IPython.core.debugger import Tracer


def resize_image(data_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files_list = os.listdir(data_path)

    for file in files_list:
        file_path = os.path.join(data_path, file)
        image = cv2.imread(file_path)
        img_h, img_w, _ = image.shape
        image_reszie = cv2.resize(image, (img_w //2, img_h//2), cv2.INTER_CUBIC)  # resize
        # cv2.imshow('image_clip', image_clip)
        cv2.imwrite(save_path + '/' + file, image_reszie)

        print('{} has been resized successfully!'.format(file))
    print('_' * 50)
    print('All files have been resized successfully!')

def clip_image(data_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files_list = os.listdir(data_path)

    for file in files_list:
        file_path = os.path.join(data_path, file)
        #Tracer()()
        image = cv2.imread(file_path)
        img_h, img_w, _ = image.shape
        #image_clip = image[200:img_h // 4, 200:img_w // 4] #图像切片
        image_clip = image[200:712, 200:712] #图像切片
        #cv2.imshow('image_clip', image_clip)
        cv2.imwrite(save_path + '/' + file, image_clip)
    
        print('{} has been cliped successfully!'.format(file))
    print('_' * 50)
    print('All files have been cliped successfully!')

def moveFiles(fileDir, tarDir):
    file_list = os.listdir(fileDir)
    files_num = len(file_list)
    rate = 0.1
    pick_num = int(files_num * rate)
    sample_blur = random.sample(file_list, pick_num)
    #sample_clear =
    print(sample_blur)
    for name in sample_blur:
        blur_data_path = os.path.join(fileDir, name)
        clear_data_path = blur_data_path.replace('blur_train_clip', 'clear_train_clip')

        save_blur_path = os.path.join(tarDir, name)
        save_clear_path = save_blur_path.replace('blur', 'clear')

        shutil.move(blur_data_path, save_blur_path)
        shutil.move(clear_data_path, save_clear_path)
        
    print('_' * 50)
    print('move files successfully!')

if __name__ == '__main__':
    data_path = 'E:/datasets/denoising/processing_data/dirty_train_0509'
    save_path = 'E:/datasets/denoising/processing_data/dirty_train_0510'

    fileDir = 'E:/datasets/denoising/processing_data/blur_train_clip'
    tarDir = 'E:/datasets/denoising/processing_data/validation/blur'

    #clip_image(data_path, save_path)
    resize_image(data_path, save_path)
    # moveFiles(fileDir, tarDir)
