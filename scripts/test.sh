#!/bin/bash
# Evaluate AVA-DINO on a single dataset
DATASET=${1:-mvtec}
CKPT_DIR=${2:-./results/${DATASET}/ckpt}
RESULT_PATH=${3:-./results_test/${DATASET}}

python test.py \
    --dataset       "${DATASET}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --result_path   "${RESULT_PATH}" \
    --device        cuda:0 \
    --batch_size    16 \
    --start_epoch   0 \
    --num_epochs    10
