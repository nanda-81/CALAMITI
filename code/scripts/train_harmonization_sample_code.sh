#!/bin/bash

python ../train_harmonization.py \
    --dataset-dirs ../../sample_dataset \
    --data-names T1 T2 \
    --orientation AXIAL CORONAL SAGITTAL \
    --epochs 5 \
    --gpu 1 \
    --batch-size 4 \
    --out-dir ../../results \
    --beta-dim 4 \
    --theta-dim 2
