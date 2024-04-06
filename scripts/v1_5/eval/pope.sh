#!/bin/bash

CKPT="multi-ve-shared-resampler-clip-dino-llava-pretrained-v1.5-7b"

python -um llava.eval.multi_ve_model_vqa_loader \
    --model-path /data/data0/akane/$CKPT/checkpoints \
    --question-file /data/data0/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /data/data0/akane/LLaVA/data/eval/pope/val2014 \
    --answers-file /data/data0/akane/LLaVA/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /data/data0/akane/LLaVA/data/eval/pope/coco \
    --question-file /data/data0/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
    --result-file /data/data0/akane/LLaVA/data/eval/pope/answers/$CKPT.jsonl
