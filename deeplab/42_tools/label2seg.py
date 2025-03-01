# -*- coding: utf-8 -*-
# @Time    : 2023/5/13 1:20
# @Author  : 肆十二
# @Email   : 3048534499@qq.com
# @File    : label2seg
# @Software: PyCharm

from __future__ import print_function
import argparse
import glob
import math
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2


def json2png(json_folder, png_save_folder, src_save_folder):
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)

    if osp.isdir(src_save_folder):
        shutil.rmtree(src_save_folder)

    os.makedirs(png_save_folder)
    json_files = os.listdir(json_folder)
    for json_file in json_files:
        json_path = osp.join(json_folder, json_file)
        os.system("labelme_json_to_dataset {}".format(json_path))
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")

        src_path = osp.join(json_folder, json_file.split(".")[0] + "_json/img.png")
        src_save_path = osp.join(src_save_folder, json_file.split(".")[0] + ".png")
        shutil.copy(src_path, src_save_path)

        label_png = cv2.imread(label_path, 0)
        img = cv2.imread(label_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_1 = img == [128, 0, 0]
        img_1 = img_1[:, :, 0] & img_1[:, :, 1] & img_1[:, :, 2]
        label_png[img_1] = 1

        img_2 = img == [0, 128, 0]
        img_2 = img_2[:, :, 0] & img_2[:, :, 1] & img_2[:, :, 2]
        label_png[img_2] = 2

        img_3 = img == [128, 128, 0]
        img_3 = img_3[:, :, 0] & img_3[:, :, 1] & img_3[:, :, 2]
        label_png[img_3] = 3

        # label_png[label_png > 0] = 255
        cv2.imwrite(png_save_path, label_png)


if __name__ == '__main__':
    # !!!!你的json文件夹下只能有json文件不能有其他文件
    json2png(json_folder="G:/money/project_dz/5_May/500_333/image",
             png_save_folder="G:/money/project_dz/5_May/500_333/labels",
             src_save_folder="G:/money/project_dz/5_May/500_333/src_imgs")
