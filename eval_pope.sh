#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=7 nohup ./scripts/v1_5/eval/pope.sh 0 >> tome_pope_evals/eval_pope_0.out &
# CUDA_VISIBLE_DEVICES=6 nohup ./scripts/v1_5/eval/pope.sh 23 >> tome_pope_evals/eval_pope_23.out &
# CUDA_VISIBLE_DEVICES=5 nohup ./scripts/v1_5/eval/pope.sh 22 >> tome_pope_evals/eval_pope_22.out &
# CUDA_VISIBLE_DEVICES=4 nohup ./scripts/v1_5/eval/pope.sh 21 >> tome_pope_evals/eval_pope_21.out &
# CUDA_VISIBLE_DEVICES=3 nohup ./scripts/v1_5/eval/pope.sh 20 >> tome_pope_evals/eval_pope_20.out &
# CUDA_VISIBLE_DEVICES=2 nohup ./scripts/v1_5/eval/pope.sh 19 >> tome_pope_evals/eval_pope_19.out

# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 17 > eval_gqa_17.out 
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 18 > eval_gqa_18.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 19 > eval_gqa_19.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 20 > eval_gqa_20.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 21 > eval_gqa_21.out 
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 22 > eval_gqa_22.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 23 > eval_gqa_23.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 24 > eval_gqa_24.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 25 > eval_gqa_25.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 26 > eval_gqa_26.out
# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./scripts/v1_5/eval/gqa.sh 16 > eval_gqa_16.out