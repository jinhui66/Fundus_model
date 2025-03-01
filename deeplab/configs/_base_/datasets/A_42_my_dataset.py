# dataset settings
dataset_type = 'MyDataset'               # 数据集类型
data_root = '../split_dataset'     # 数据集的根目录
# reduce_zero_label=True                 # 是否减少标签中的零值，通常用于忽略背景
is_keep_ratio = False                    # 是否保持图像的长宽比例
# crop_size = (640, 640)                 # 裁剪尺寸
global_size = 512                        # 全局尺寸
scale_size = (global_size, global_size)  # 缩放尺寸
crop_size = (global_size, global_size)   # 裁剪尺寸

# 训练数据的处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),   # 从文件中加载图像
    dict(type='LoadAnnotations'),     # 加载标注（分割标签）
    # dict(type='LoadAnnotations'),   # 冗余的加载标注操作，已注释掉
    dict(
        type='RandomResize',          # 随机调整图像尺寸
        scale=scale_size,             # 调整后的尺寸
        ratio_range=(0.5, 2.0),       # 调整比例范围
        keep_ratio=is_keep_ratio),    # 是否保持长宽比例
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪图像
    dict(type='RandomFlip', prob=0.5),     # 随机翻转图像
    # dict(type='PhotoMetricDistortion'),  # 光度失真（增强图像），已注释掉
    dict(type='PackSegInputs')             # 打包分割输入
]

# 测试数据的处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),                                   # 从文件中加载图像
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),        # 调整图像尺寸（已注释掉）
    dict(type='Resize', scale=scale_size, keep_ratio=is_keep_ratio),  # 调整图像尺寸
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),                                     # 加载标注（通常测试时不用调整大小）
    dict(type='PackSegInputs')                                        # 打包分割输入
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_ratios = [1.0,]                                     # 图片缩放比例
# TTA (Test Time Augmentation) 流水线
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),  # 从文件中加载图像
    dict(
        type='TestTimeAug',  # 测试时增强
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=is_keep_ratio)
                for r in img_ratios                                         # 使用不同的缩放比例进行调整
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),    # 不翻转
                dict(type='RandomFlip', prob=1., direction='horizontal')     # 水平翻转
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]  # 加载标注
        ])
]
# 训练数据加载器配置
train_dataloader = dict(
    batch_size=2,                                        # 每批次的样本数量
    num_workers=4,                                       # 使用的工作进程数
    persistent_workers=True,                             # 工作进程是否持久化
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 采样器配置，使用无限采样并打乱顺序
    dataset=dict(
        type=dataset_type,                               # 数据集类型
        data_root=data_root,                             # 数据集根目录
        data_prefix=dict(
            img_path='Training_Images', seg_map_path='Training_Labels'),  # 图像和标签路径前缀
        pipeline=train_pipeline))                                         # 训练数据的处理流水线
# 验证数据加载器配置
val_dataloader = dict(
    batch_size=1,                                        # 每批次的样本数量
    num_workers=1,                                       # 使用的工作进程数
    persistent_workers=True,                             # 工作进程是否持久化
    sampler=dict(type='DefaultSampler', shuffle=False),  # 采样器配置，使用默认采样器并不打乱顺序
    dataset=dict(
        type=dataset_type, # 数据集类型
        data_root=data_root, # 数据集根目录
        data_prefix=dict(
            # img_path='images/validation',                   # 验证图像路径（已注释掉）
            img_path='Test_Images',                           # 验证图像路径
            seg_map_path='Test_Labels'),                      # 验证标签路径
        pipeline=test_pipeline))                              # 验证数据的处理流水线
# 测试数据加载器配置，与验证数据加载器相同
test_dataloader = val_dataloader
# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # 使用IoU作为评估指标
test_evaluator = val_evaluator                                # 测试评估器与验证评估器相同

