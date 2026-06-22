_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 20
num_training_classes = 20
ori_num_classes = 10
max_epochs = 20  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 12
load_from = './weights/dior_10_10_t1.pth'  # will be set dynamically or ignored
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False

classes = [
    "airplane", "airport", "bridge", "Expressway-Service-area", "Expressway-toll-station", 
    "harbor", "overpass", "ship", "trainstation", "vehicle",
    "baseballfield", "basketballcourt", "chimney", "dam", "golffield", 
    "groundtrackfield", "stadium", "storagetank", "tenniscourt", "windmill"
]

# model settings
model = dict(
    type='YOLOIODDetector',
    load_from_weight=load_from,
    ori_setting=dict(
        config='configs/dior_10_10/yolo_iod_dior_10_10_task0.py',
        ckpt='work_dirs/yolo_iod_dior_10_10_task0/epoch_20.pth',
    ),
    cur_setting=dict(
        config='configs/dior_10_10/yolo_iod_dior_10_10_stage1.py',
        ckpt='work_dirs/yolo_iod_dior_10_10_stage1/epoch_20.pth',
    ),
    kd_cfg=dict(
        loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),
        loss_reg_kd=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_cls_kd_temperature_old=1000,
        loss_reg_kd_temperature_old=2,
        loss_cls_kd_temperature_new=800,
        loss_reg_kd_temperature_new=1,
        kd_new=True,
        class_names=classes,
        ori_num_classes=ori_num_classes
    ),
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCacheCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldCrossKdScoreHead',
                   head_module=dict(type='YOLOWorldCrossKdScoreHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes, type='BatchTaskAlignedScoreV2Assigner')))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmyolo.PackDetInputsScore',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts', 'gt_bboxes_scores'))
]
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]
dior_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoScoreDataset',
        data_root='/content/data/DIOR/',
        metainfo=dict(classes=classes),
        ann_file='annotation/annotation/train_task_2_ps.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/content/data/DIOR/annotation/annotation/dior_class_texts_stage1.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate_score'),
    dataset=dior_train_dataset)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

dior_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/content/data/DIOR/',
        ann_file='annotation/annotation/test_task_12.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/content/data/DIOR/annotation/annotation/dior_class_texts_stage1.json',
    pipeline=test_pipeline)
val_dataloader = dict(
    batch_size=8,
    dataset=dior_val_dataset,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataloader = val_dataloader
# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=4,
        save_best=None,
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainGPSLoop',
    gps_ratio=0.8,
    up_ratio=0.8,
    importance_save_path='/content/data/DIOR/annotation/annotation/t1_importance_iod.pt',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(3, 5), ((max_epochs - close_mosaic_epochs), 3)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    classwise=True,
    metric_items=['mAP', 'mAP_50', 'mAP_75'],
    ann_file='/content/data/DIOR/annotation/annotation/test_task_12.json',
    metric='bbox')
test_evaluator = val_evaluator
find_unused_parameters = True
