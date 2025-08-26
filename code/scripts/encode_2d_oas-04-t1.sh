#!/bin/bash

python ../encode_2d.py \
    --in-dir /iacl/pg20/lianrui/projects/disentangled_harmonization/data/oasis3/travel_subjs/slices/scan-04/ \
    --data-name T1 \
    --orientation AXIAL \
    --out-dir ../../tests/oas-ixi-blsa/encode_2d/oas-04 \
    --pretrained-model ../../tests/oas-ixi-blsa/models/epoch009_batch2000.pt \
    --gpu 1 \
    --theta-dim 2 \
    --beta-dim 4
