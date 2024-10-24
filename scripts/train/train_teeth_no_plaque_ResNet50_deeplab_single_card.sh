#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/ResNet50_FCN_m4_decouple_ft_175_e
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --dataset teeth_no_plaque \
  --cv 0 \
  --arch network.deepv3.DeepR50V3PlusD_m1_deeply \
  --class_uniform_pct 0.0 \
  --class_uniform_tile 1024 \
  --max_cu_epoch 70 \
  --lr 0.001 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --sgd \
  --crop_size 700 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 70 \
  --ohem \
  --ohem \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --apex \
  --exp teeth_no_plaque_ft_607 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
