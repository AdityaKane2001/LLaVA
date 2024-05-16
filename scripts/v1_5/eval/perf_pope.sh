#!/bin/bash
CKPT="liuhaotian/llava-v1.5-7b"

python -um llava.eval.model_perf \
    --model-path $CKPT \
    --question-file /data/data0/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /data/data0/akane/LLaVA/data/eval/pope/val2014 \
    --answers-file /data/data0/akane/LLaVA/data/eval/pope/answers/${CKPT}_r_${1}.jsonl \
    --r $1\
    --temperature 0 \
    --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir  /data/data0/akane/LLaVA/data/eval/pope/coco \
#     --question-file /data/data0/akane/LLaVA/data/eval/pope/llava_pope_test.jsonl \
#     --result-file /data/data0/akane/LLaVA/data/eval/pope/answers/${CKPT}_r_${1}.jsonl
