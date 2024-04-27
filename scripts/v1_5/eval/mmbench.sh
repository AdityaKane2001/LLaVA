#!/bin/bash

SPLIT="mmbench_dev_en_20231003"
CKPT="mve-clip-dino-router-finetune"
#multi_ve_model_vqa_loader
python -um llava.eval.multi_ve_model_vqa_mmbench \
    --model-path /data/data1/akane/$CKPT/checkpoints \
    --question-file /data/data1/akane/LLaVA/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /data/data1/akane/LLaVA/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /data/data1/akane/LLaVA/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /data/data1/akane/LLaVA/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /data/data1/akane/LLaVA/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT