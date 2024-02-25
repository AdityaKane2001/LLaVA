#!/bin/bash

MODEL_NAME="dupl-glbltok-grllava-v1.5-7b"

python -m llava.eval.model_vqa_loader \
    --model-path /data/data0/akane/dupl-glbltok-grllava-v1.5-7b/checkpoints \
    --question-file /data/data1/akane/LLaVA/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /data/data1/akane/LLaVA/data/eval/vizwiz/test \
    --answers-file /data/data1/akane/LLaVA/data/eval/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /data/data1/akane/LLaVA/data/eval/vizwiz/llava_test.jsonl \
    --result-file /data/data1/akane/LLaVA/data/eval/vizwiz/answers/$MODEL_NAME.jsonl \
    --result-upload-file /data/data1/akane/LLaVA/data/eval/vizwiz/answers_upload/$MODEL_NAME.json
