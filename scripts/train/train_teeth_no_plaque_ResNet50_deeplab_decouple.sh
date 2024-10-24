#!/usr/bin/env bash
rm ./teeth_train_cv0_tile1024.json
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/ResNet50_FCN_m4_decouple_ft_175_e
mkdir -p ${EXP_DIR}
python -m torch.distributed.launch --nproc_per_node=2 train.py \
  --dataset teeth_no_plaque \
  --cv 0 \
  --snapshot ./pretrained_models/best_epoch_62_mean-iu_0.94141_610_teeth_no_plaque.pth \
  --arch network.deepv3_decouple.DeepR50V3PlusD_m1_deeply \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --max_cu_epoch 150 \
  --lr 0.0005 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 700 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 175 \
  --ohem \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --apex \
  --exp teeth_ft_611_teeth_no_plaque \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &
