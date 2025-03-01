# optimizer
optimizer = dict(type='SGD',                          # 优化器类型为随机梯度下降（SGD）
                 lr=0.01/4,                           # 学习率为0.01的四分之一（可能是因为多卡训练，进行了学习率缩放）
                 momentum=0.9,                        # 动量参数，用于加速收敛
                 weight_decay=0.0005)                 # 权重衰减，防止过拟合
optim_wrapper = dict(type='OptimWrapper',             # 优化器包装器，用于统一管理优化器配置
                     optimizer=optimizer,             # 使用上面定义的优化器
                     clip_grad=None)                  # 梯度裁剪，默认不裁剪
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',                                # 学习率调度策略为多项式衰减（Polynomial Decay）
        eta_min=1e-4,                                 # 最小学习率为1e-4
        power=0.9,                                    # 多项式衰减的指数
        begin=0,                                      # 开始衰减的迭代次数
        end=40000,                                    # 结束衰减的迭代次数
        by_epoch=False)                               # 按迭代次数而不是按周期进行调度
]
# training schedule for 40k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
train_cfg = dict(type='IterBasedTrainLoop',           # 基于迭代次数的训练循环
                 max_iters=40000,                     # 最大训练迭代次数为40000次
                 val_interval=20000)                  # 每20000次迭代进行一次验证
                 # val_interval=100)                  # 每20000次迭代进行一次验证
val_cfg = dict(type='ValLoop')                        # 验证循环配置
test_cfg = dict(type='TestLoop')                      # 测试循环配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),                 # 计时器钩子，用于记录每次迭代的时间
    logger=dict(type='LoggerHook',                    # 日志钩子，用于记录训练过程中的日志
                interval=50,                          # 每50次迭代记录一次日志
                log_metric_by_epoch=False),           # 不按周期记录指标，按迭代记录
    param_scheduler=dict(type='ParamSchedulerHook'),  # 学习率调度钩子
    checkpoint=dict(type='CheckpointHook',            # 检查点钩子，用于保存模型
                    by_epoch=False,                   # 不按周期保存，按迭代次数保存
                    interval=20000),                  # 每20000次迭代保存一次模型
    sampler_seed=dict(type='DistSamplerSeedHook'),    # 分布式采样种子钩子
    visualization=dict(type='SegVisualizationHook'))  # 分割结果可视化钩子
