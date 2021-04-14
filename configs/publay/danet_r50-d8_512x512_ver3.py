norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DAHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pam_channels=16,
        dropout_ratio=0.1,
        num_classes=16,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=16,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(704, 704), ratio_range=None),
    # dict(type='RandomCrop', crop_size=(704, 704), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='Pad', size=(704, 704), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(704, 704),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='DRIVEDataset',
        img_dir='/netscratch/minouei/versicherung/version3/images/train',
        ann_dir='/netscratch/minouei/versicherung/version3/annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type='DRIVEDataset',
        img_dir='/netscratch/minouei/versicherung/version3/images/val',
        ann_dir='/netscratch/minouei/versicherung/version3/annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type='DRIVEDataset',
        img_dir='/netscratch/minouei/versicherung/version3/images/val',
        ann_dir='/netscratch/minouei/versicherung/version3/annotations/val',
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
work_dir = '/netscratch/minouei/versicherung/work_dirs/danet_r18-d8_704x704_80k_ver3'
gpu_ids = range(0, 1)
