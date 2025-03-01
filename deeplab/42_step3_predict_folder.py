#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：42_predict_image.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：文件夹结果批量预测
@Date    ：2024/8/27 16:24 
'''
import os

import cv2

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：42_demo.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：用了批量预测一个文件夹里面的图像，需要保证你得原始文件夹里面只有图像文件，不能有其他的文件
@Date    ：2024/8/27 16:24 
'''
import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os.path as osp
import time

config_file = 'work_dirs/refuge-deeplabv3plus_r50-d8_4xb2-40k_512x512/refuge-deeplabv3plus_r50-d8_4xb2-40k_512x512.py'   # 模型配置文件路径
checkpoint_file = 'work_dirs/refuge-deeplabv3plus_r50-d8_4xb2-40k_512x512/iter_40000.pth'                            # 模型路径
# SRC_FOLDER = "/data3/wangchangmiao/jinhui/eye/Enhanced"   
SRC_FOLDER = "/data3/wangchangmiao/jinhui/eye/ietk_Enhanced"# 需要进行预测的文件夹路径
SAVE_FOLDER = "/data3/wangchangmiao/jinhui/eye/result_ietk_seg"                                     # 预测结果保存的路径

def predict_folder(src_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)                        # 创建要进行保存的文件夹
    # model = init_model(config_file, checkpoint_file, device='cpu') # 模型初始化，模型初始化在CPU上面
    model = init_model(config_file, checkpoint_file, device='cuda:1') # 模型初始化，模型初始化在CPU上面
    if not torch.cuda.is_available():                              # 如果GPU是可用的，则可以直接在GPU上对模型执行实例化
        model = revert_sync_batchnorm(model)
    model.dataset_meta['classes'] = ('background', ' Optic Cup', 'Optic Disc')
    model.dataset_meta['palette'] = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]

    image_names = os.listdir(src_folder)                           # 获取所有的图像名称
    total_num = len(image_names)
    time_start = time.time()  # 记录开始时间


    for i, image_name in enumerate(image_names):                                 # 循环处理每一张图像
        print(f"{i+1}/{total_num}:{image_name}")
        src_image_path = osp.join(src_folder, image_name)
        save_image_path = osp.join(save_folder, image_name)
        result = inference_model(model, src_image_path)
        vis_result = show_result_pyplot(model, src_image_path, result, show=False, with_labels=False, opacity=1.0)
        save_img = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_image_path, save_img)
        # vis_result.save(save_image_path, quality=100)
        # --------------------------------------------------------------------------------
        # save方法参数解析
        # 参数解释
        # file_path:图片输出位置
        # dpi:设置图像的 DPI 是 (horizontal_dpi, vertical_dpi) 水平和竖直DPI为300
        # quality（可选参数）：用于指定 JPEG 格式的图像质量，取值范围为 1-95。实际调用时都是些100
        # optimize（可选参数）：用于某些格式（如 GIF），指定是否优化保存的图像文件大小，默认为False。如果设置为True，则会尝试减小文件大小，但可能会增加保存时间。
        # icc_profile：用于指定图像的 ICC（International Color Consortium）配置文件的路径。
        # --------------------------------------------------------------------------------
        # 将图像直接进行保存，按照pillow的格式
        # 是否进行图像的展示
        # plt.imshow(vis_result)
        # plt.show()
        # function()   执行的程序
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(f"预测{total_num}张图片的总耗时为{time_sum}s，该模型的fps为{total_num/time_sum}")

if __name__ == '__main__':
    predict_folder(src_folder=SRC_FOLDER, save_folder=SAVE_FOLDER)