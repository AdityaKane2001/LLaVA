#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
CKPT="liuhaotian/llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/data/data0/akane/LLaVA/data/eval/gqa/data"

echo $0 $1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -um llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file /data/data0/akane/LLaVA/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /data/data0/akane/LLaVA/data/eval/gqa/data/images \
        --answers-file /data/data0/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --r $1\
        --conv-mode vicuna_v1 &
done

wait

output_file=/data/data0/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/data0/akane/LLaVA/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced