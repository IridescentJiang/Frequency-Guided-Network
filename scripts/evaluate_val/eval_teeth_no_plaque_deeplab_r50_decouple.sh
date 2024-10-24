#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
  python3 eval.py \
	--dataset teeth_no_plaque \
    --arch network.deepv3_decouple.DeepR50V3PlusD_m1_deeply \
    --inference_mode  sliding \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --crop_size 700 \
    --ckpt_path ${2} \
    --snapshot ${1}
