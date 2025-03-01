# model settings
norm_cfg = dict(type='BN', requires_grad=True)  # 归一化配置，使用批归一化（BN），并且允许梯度更新
data_preprocessor = dict(
    type='SegDataPreProcessor',                 # 数据预处理器的类型为SegDataPreProcessor
    mean=[123.675, 116.28, 103.53],             # 图像的均值，用于归一化
    std=[58.395, 57.12, 57.375],                # 图像的标准差，用于归一化
    bgr_to_rgb=True,                            # 是否将BGR图像转换为RGB图像
    pad_val=0,                                  # 填充的数值（用于图像）
    seg_pad_val=255)                            # 填充的数值（用于分割标签）
model = dict(
    type='EncoderDecoder',                      # 模型类型为Encoder-Decoder架构
    data_preprocessor=data_preprocessor,        # 使用上面定义的数据预处理器
    pretrained='open-mmlab://resnet50_v1c',     # 预训练模型，使用ResNet50 v1c
    backbone=dict(
        type='ResNetV1c',                       # 主干网络类型为ResNetV1c
        depth=50,                               # ResNet的深度为50层
        num_stages=4,                           # ResNet的阶段数为4
        out_indices=(0, 1, 2, 3),               # 输出每个阶段的特征图索引
        dilations=(1, 1, 2, 4),                 # 各阶段的膨胀率
        strides=(1, 2, 1, 1),                   # 各阶段的步幅
        norm_cfg=norm_cfg,                      # 归一化配置
        norm_eval=False,                        # 是否在测试时冻结BN层
        style='pytorch',                        # 使用PyTorch的卷积风格
        contract_dilation=True),                # 是否在膨胀卷积中收缩膨胀率
# 解码头的设置
    decode_head=dict(
        type='ASPPHead',                        # 解码头的类型为ASPP（空洞空间金字塔池化）头
        in_channels=2048,                       # 输入通道数为2048
        in_index=3,                             # 输入特征图索引
        channels=512,                           # 中间通道数为512
        dilations=(1, 12, 24, 36),              # 空洞率配置
        dropout_ratio=0.1,                      # Dropout比率
        num_classes=19,                         # 输出的类别数量为19
        norm_cfg=norm_cfg,                      # 归一化配置
        align_corners=False,                    # 是否对齐角点，用于上采样, 指的是图像四个边界的点是否严格对齐，不对齐更符合常理
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),                  # 解码损失函数使用交叉熵损失
    # 辅助头的设置（用于辅助训练）
    auxiliary_head=dict(
        type='FCNHead',                         # 辅助头的类型为全卷积网络（FCN）头
        in_channels=1024,                       # 输入通道数为1024
        in_index=2,                             # 输入特征图索引
        channels=256,                           # 中间通道数为256
        num_convs=1,                            # 卷积层数为1
        concat_input=False,                     # 是否将输入特征与输出特征拼接
        dropout_ratio=0.1,                      # Dropout比率
        num_classes=19,                         # 输出的类别数量为19
        norm_cfg=norm_cfg,                      # 归一化配置
        align_corners=False,                    # 是否对齐角点，用于上采样
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.4)),                 # 辅助头的损失函数使用交叉熵损失，权重为0.4
    # model training and testing settings
    # 模型训练和测试设置
    train_cfg=dict(),                          # 训练配置（可为空）
    test_cfg=dict(mode='whole'))               # 测试配置，使用整体测试模式
