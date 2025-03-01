# -*- coding:utf-8 -*-
# @Time: 2022/11/4 10:25
# @Author: ChenmingSong
# @Email: SongCM@CATL.com
# @File: cell_data.py
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    """STARE dataset.
    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    # 颜色变化表格，其中这里是RGB格式的，如果您需要和数据集中的对应，下面的palette请按照BGR的格式填写，也就是对调第一个位置和第三个位置
    # [[  0   0   0]
    #  [128   0   0]
    #  [  0 128   0]
    #  [128 128   0]
    #  [  0   0 128]
    #  [128   0 128]
    #  [  0 128 128]
    #  [128 128 128]
    #  [ 64   0   0]
    #  [192   0   0]
    #  [ 64 128   0]
    #  [192 128   0]
    #  [ 64   0 128]
    #  [192   0 128]
    #  [ 64 128 128]
    #  [192 128 128]
    #  [  0  64   0]
    #  [128  64   0]
    #  [  0 192   0]
    #  [128 192   0]
    # classes=('background', ' Optic Cup', 'Optic Disc'),
    # 将颜色进行纠正
    METAINFO = dict(
        # 分别表示的是背景区域、视杯区域和视盘区域
        classes=('background', ' Optic Disc', 'Optic Cup'),
        # 涂色盘，即模型预测出来之后是如何给预测结果图像上色的
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0]])
        # palette=[[0, 0, 0], [1, 1, 1], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        # assert self.file_client.exists(self.data_prefix['img_path'])
