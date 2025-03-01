import os
import numpy as np
import cv2
import copy
src_label_folder = "../../../eye-3-classes/Training_Labels_src"
save_folder = src_label_folder + "_seg"
if os.path.isdir(save_folder):
    pass
else:
    os.mkdir(save_folder)
labels = os.listdir(src_label_folder)
for label in labels:
    print(label)
    label_path = os.path.join(src_label_folder, label)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # print(label_img)
    # 这样转化之后导致面积计算出错
    # 应该是直接计算
    label_img_copy = copy.deepcopy(label_img)
    label_img_copy[label_img == 255] = 0
    label_img_copy[label_img == 0] = 1
    label_img_copy[label_img == 128] = 2
    save_path = os.path.join(save_folder, label)
    cv2.imwrite(save_path, label_img_copy)
    # break
