#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：42_predict_video.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/27 16:24 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：read_webcam.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：用来做视频文件的预测
@Date    ：2024/9/10 18:03 
'''
import cv2
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os.path as osp
import time
import torch

fps = 20
config_file = 'work_dirs/refuge-deeplabv3_r18-d8_4xb2-20k-512x512/refuge-deeplabv3_r18-d8_4xb2-20k-512x512.py'   # 模型配置文件路径
checkpoint_file = 'work_dirs/refuge-deeplabv3_r18-d8_4xb2-20k-512x512/iter_40000.pth'                            # 模型路径
SRC_VIDEO_PATH = "demo/output.mp4"
TARGET_SAVE_PATH = "demo/output-result.mp4"

# 模型加载
model = init_model(config_file, checkpoint_file, device='cuda:0')  # 模型初始化，模型初始化在CPU上面
if not torch.cuda.is_available():  # 如果GPU是可用的，则可以直接在GPU上对模型执行实例化
    model = revert_sync_batchnorm(model)

model.dataset_meta['classes'] = ('background', ' Optic Cup', 'Optic Disc')
model.dataset_meta['palette'] = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]


# 定义视频捕获对象，0代表第一个摄像头
cap = cv2.VideoCapture(SRC_VIDEO_PATH)
# 定义视频编解码器和创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
# 实现读取一帧 用来获取当前需要进行保存的画面大小
ret, frame = cap.read()  # 从摄像头读取一帧
height, width, layers = frame.shape
out = cv2.VideoWriter(TARGET_SAVE_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
result = inference_model(model, frame)
vis_result = show_result_pyplot(model, frame, result, show=False, with_labels=True, opacity=1.0)
save_img = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
# 显示帧
cv2.imshow('frame', save_img)
out.write(save_img)
# out.write(frame)

while cap.isOpened():
    ret, frame = cap.read()  # 从摄像头读取一帧
    if ret:
        # 将帧写入文件
        # out.write(frame)
        # 对当前的图像进行处理并显示结果
        result = inference_model(model, frame)
        vis_result = show_result_pyplot(model, frame, result, show=False, with_labels=True, opacity=1.0)
        save_img = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)

        # 显示帧
        cv2.imshow('frame', save_img)
        out.write(save_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()