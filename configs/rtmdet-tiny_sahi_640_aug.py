auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.005
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
classes = 'defective_sunflower'
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=250,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    640,
                    640,
                ),
                type='RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
]
data_root = 'data/sunflower_dataset_v2.0_sliced_640'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=20,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
double_scale = (
    1280,
    1280,
)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_scale = (
    640,
    640,
)
interval = 10
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(
            2,
            3,
            4,
        ),
        type='CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=False,
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.5,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=2,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=96,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.005, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=150,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.00025,
        type='CosineAnnealingLR'),
]
resume = False
stage2_num_epochs = 50
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val/val-annotations.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/sunflower_dataset_v2.0_sliced_640',
        metainfo=dict(classes=['defective_sunflower', 'healthy_sunflower']),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/sunflower_dataset_v2.0_sliced_640/val/val-annotations.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        3000,
        3000,
        3000,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='train/train-annotations.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/sunflower_dataset_v2.0_sliced_640',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=['defective_sunflower', 'healthy_sunflower']),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='CachedMosaic'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    1280,
                    1280,
                ),
                type='RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=20,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline_aug1 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='CachedMosaic'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1280,
            1280,
        ),
        type='RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=20,
        pad_val=(
            114,
            114,
            114,
        ),
        ratio_range=(
            1.0,
            1.0,
        ),
        type='CachedMixUp'),
    dict(type='PackDetInputs'),
]
train_pipeline_aug2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
train_pipeline_noaug = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val/val-annotations.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/sunflower_dataset_v2.0_sliced_640',
        metainfo=dict(classes=['defective_sunflower', 'healthy_sunflower']),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/sunflower_dataset_v2.0_sliced_640/val/val-annotations.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        3000,
        3000,
        3000,
    ),
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './training-output/rtmdet-tiny_sahi_640_aug'
