#!/bin/bash
# Train AVA-DINO on MVTec-AD
python train.py \
    --dataset mvtec \
    --result_path ./results/mvtec \
    --device cuda:0 \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4
