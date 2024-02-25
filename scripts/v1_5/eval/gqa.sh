#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
CKPT="dupltok-llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/data/data1/akane/LLaVA/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -um llava.eval.model_vqa_loader \
        --model-path /data/data0/akane/dupl-glbltok-grllava-v1.5-7b/checkpoints \
        --question-file /data/data1/akane/LLaVA/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /data/data1/akane/LLaVA/data/eval/gqa/data/images \
        --answers-file /data/data1/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/data/data1/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/data1/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
