#!/bin/bash

SPLIT="mmbench_dev_en_20231003"
CKPT="dupl-glbltok-grllava-v1.5-7b"
 #/data/data1/akane/grllava-v1.5-7b/checkpoints \
python -m llava.eval.model_vqa_mmbench \
    --model-path /data/data0/akane/dupl-glbltok-grllava-v1.5-7b/checkpoints \
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
