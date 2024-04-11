#!/bin/bash

CKPT="mve-clip-dino-finetune-2"

python -um llava.eval.multi_ve_model_vqa_loader \
    --model-path /data/data1/akane/$CKPT/checkpoints \
    --question-file /data/data1/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /data/data1/akane/LLaVA/data/eval/pope/val2014 \
    --answers-file /data/data1/akane/LLaVA/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /data/data1/akane/LLaVA/data/eval/pope/coco \
    --question-file /data/data1/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
    --result-file /data/data1/akane/LLaVA/data/eval/pope/answers/$CKPT.jsonl
