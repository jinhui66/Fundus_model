#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：42_train.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/27 16:23 
'''
import os
os.system("python tools/train.py configs/A_42_eye/refuge-deeplabv3_r18-d8_4xb2-20k-512x512.py")
os.system("python tools/train.py configs/A_42_eye/refuge-deeplabv3_r50-d8_4xb2-20k-512x512.py")
os.system("python tools/train.py configs/A_42_eye/refuge-deeplabv3plus_r18-d8_4xb2-40k_512x512.py")
os.system("python tools/train.py configs/A_42_eye/refuge-deeplabv3plus_r50-d8_4xb2-40k_512x512.py")