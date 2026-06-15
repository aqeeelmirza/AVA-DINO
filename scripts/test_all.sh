#!/bin/bash
# Evaluate AVA-DINO across all supported datasets
CKPT_DIR=${1:-./results/mvtec/ckpt}

DATASETS=(mvtec visa btad mpdd mvtecad2 clinicdb colondb kvasirseg br35h brainmri headct isic ksdd2 tn3k)

for DATASET in "${DATASETS[@]}"; do
    echo "========================================="
    echo "Evaluating: ${DATASET}"
    echo "========================================="
    python test.py \
        --dataset        "${DATASET}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --result_path    "./results_test/${DATASET}" \
        --device         cuda:0 \
        --batch_size     16 \
        --start_epoch    0 \
        --num_epochs     10
done
