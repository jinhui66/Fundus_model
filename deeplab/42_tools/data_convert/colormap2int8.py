#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：colormap2int8.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/9/6 22:05 
'''
import os

import cv2
from PIL import Image
import numpy as np
image_src_folder = r"F:\AAA-projects\ING\999\300-airport\split_dataset\train_labels_1920x1080"
image_save_folder = r"F:\AAA-projects\ING\999\300-airport\split_dataset\train_labels_1920x1080_int"
os.makedirs(image_save_folder, exist_ok=True)
for name in os.listdir(image_src_folder):
    print(name)
    src_image_path = os.path.join(image_src_folder, name)
    save_image_path = os.path.join(image_save_folder, name)
    img = Image.open(src_image_path)
    img = np.array(img, dtype=np.uint8)
    # print(img.shape)
    cv2.imwrite(save_image_path, img)