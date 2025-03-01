# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 0:43
# @Author  : 肆十二
# @Email   : 3048534499@qq.com
# @File    : rename
# @Software: PyCharm
import os
import os.path as osp
src_folder = r"E:\danziiiiiiii\ccccccccccc\300_sar\go"
image_names = os.listdir(src_folder)
for image_name in image_names:
    image_path = osp.join(src_folder, image_name)
    os.rename(image_path, image_path.replace("Label", "SAR"))