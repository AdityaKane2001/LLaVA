#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo $0 $1

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file /data/data0/akane/LLaVA/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /data/data0/akane/LLaVA/data/eval/vqav2/test2015 \
        --answers-file /data/data0/akane/LLaVA/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_r${1}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --r $1 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/data/data0/akane/LLaVA/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/data0/akane/LLaVA/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_r${1}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir "/data/data0/akane/LLaVA/data/eval/vqav2" --split $SPLIT --ckpt ${CKPT}_r${1}


# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path liuhaotian/llava-v1.5-13b \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

