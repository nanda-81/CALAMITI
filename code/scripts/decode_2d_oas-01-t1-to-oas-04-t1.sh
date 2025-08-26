#!/bin/bash

python ../decode_2d.py \
    --in-beta-dir ../../tests/oas-ixi-blsa/encode_2d/oas-01 \
    --in-theta ../../tests/oas-ixi-blsa/encode_2d/theta/OAS-04-T1-THETA.txt \
    --data-name T1 \
    --out-dir ../../tests/oas-ixi-blsa/decode_2d/oas-01-t1-to-oas-04-t1 \
    --pretrained-model ../../tests/oas-ixi-blsa/models/epoch009_batch2000.pt \
    --gpu 1 \
    --beta-dim 4 \
    --theta-dim 2 
