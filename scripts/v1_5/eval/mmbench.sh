#!/bin/bash

CKPT="llava-v1.5-7b"
SPLIT="mmbench_dev_en_20231003"

echo $0 $1

python -m llava.eval.model_vqa_mmbench \
    --model-path "liuhaotian/llava-v1.5-7b" \
    --question-file /data/data0/akane/LLaVA/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /data/data0/akane/LLaVA/data/eval/mmbench/answers/$SPLIT/${CKPT}_r_${1}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --r $1\
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /data/data0/akane/LLaVA/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /data/data0/akane/LLaVA/data/eval/mmbench/answers/$SPLIT/liuhaotian \
    --upload-dir /data/data0/akane/LLaVA/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT}_r_${1}