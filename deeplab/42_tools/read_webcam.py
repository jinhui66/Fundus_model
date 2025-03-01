#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：read_webcam.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/9/10 18:03 
'''
import cv2

# 定义视频捕获对象，0代表第一个摄像头
cap = cv2.VideoCapture("output.mp4")

# 定义视频编解码器和创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
fps = 20
# 实现读取一帧 用来获取当前需要进行保存的画面大小
ret, frame = cap.read()  # 从摄像头读取一帧
height, width, layers = frame.shape
out = cv2.VideoWriter("output-result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
out.write(frame)

while cap.isOpened():
    ret, frame = cap.read()  # 从摄像头读取一帧
    if ret:
        # 将帧写入文件
        out.write(frame)
        # 显示帧
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 按'q'键退出
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()