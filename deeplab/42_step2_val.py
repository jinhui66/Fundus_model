#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmsegmentation-main 
@File    ：42_val.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：用于模型本地的验证测试
@Date    ：2024/8/27 16:23 
'''
import os
os.system("python tools/test.py work_dirs/refuge-deeplabv3_r18-d8_4xb2-20k-512x512/refuge-deeplabv3_r18-d8_4xb2-20k-512x512.py work_dirs/refuge-deeplabv3_r18-d8_4xb2-20k-512x512/iter_40000.pth")
