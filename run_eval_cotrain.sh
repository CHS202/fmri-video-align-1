#!/bin/bash

# python eval_cotrain.py \
#     --model resnet_18 \
#     --task annot-reg \
#     --approach not-cotrain \
#     --splits 1 2 3 4 \
#     --subs 1 2 3 4 5 6 7

python eval_cotrain.py \
    --model shufflenet_v1 \
    --task design \
    --approach cotrain \
    --splits 1 2 3 4 \
    --subs 01 02 03 06 08 09 12

python eval_cotrain.py \
    --model video_swin \
    --task design \
    --approach cotrain \
    --splits 1 2 3 4 \
    --subs 01 02 03 06 08 09 12

python eval_cotrain.py \
    --model resnet_18 \
    --task design \
    --approach cotrain \
    --splits 1 2 3 4 \
    --subs 01 02 03 06 08 09 12

# python eval_cotrain.py \
#     --model vit_3d \
#     --task annot-reg \
#     --approach not-cotrain \
#     --splits 1 2 3 4 \
#     --subs 1 2 3 4 5 6 7

python eval_cotrain.py \
    --model vit_3d \
    --task design \
    --approach cotrain \
    --splits 1 2 3 4 \
    --subs 01 02 03 06 08 09 12


