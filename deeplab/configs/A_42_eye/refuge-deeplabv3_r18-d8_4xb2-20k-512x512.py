_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/A_42_my_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model = dict(
#     pretrained='open-mmlab://resnet18_v1c',
#     backbone=dict(depth=18),
#     decode_head=dict(
#         in_channels=512,
#         channels=128,
#     ),
#     auxiliary_head=dict(in_channels=256, channels=64))
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=512,
    channels=128,
    num_classes=3),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=3))

# 可视化操作
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')