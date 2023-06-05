#!/usr/bin/env bash
EXPT_NAME=siesta_imagenet_experiment
IMAGE_DIR=/data/datasets/ImageNet1K
LABEL_ORDER_DIR=./imagenet_files/
BASE_INIT_CKPT=./swav_100c_2000e_mobilenet_modified_gelu_updated.pth
GPU=0
MAX_BUFFER_SIZE=959665 # budget=1.51 Gigabytes
CODEBOOK_SIZE=256
NUM_CODEBOOKS=8
BASE_INIT_CLASSES=100
STREAMING_MIN_CLASS=100
CLASS_INCREMENT=50 #25
NUM_CLASSES=1000
EPOCH=50
BATCH=512 # min-batch size during sleep
BS=256 # data loader for other purpose
RESUME=cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_exp.py \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${STREAMING_MIN_CLASS} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier_ckpt ${BASE_INIT_CKPT} \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--sup_epoch ${EPOCH} \
--step_size 15 \
--lr_gamma 0.1 \
--sleep_batch_size ${BATCH} \
--batch_size ${BS} \
--weight_decay 1e-5 \
--init_lr 1.6 \
--base_arch MobNetClassifyAfterLayer8 \
--classifier MobNet_StartAt_Layer8 \
--extract_features_from model.features.7 \
--num_channels 80 \
--spatial_feat_dim 14 \
--penul_feat_dim 1280 \
--resume_full_path ${RESUME} \
--save_dir ${EXPT_NAME} \
--expt_name ${EXPT_NAME} \
--ckpt_file ${EXPT_NAME}.pth > logs/${EXPT_NAME}.log
