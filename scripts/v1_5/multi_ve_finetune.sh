#!/bin/bash

echo "\n================================================= $(date -u) =========================================================\n"
export WANDB_PROJECT="multi-ve-llava"
deepspeed llava/train/multi_ve_train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/data1/akane/LLaVA/data/llava_v1_5_mix665k.json \
    --image_folder /data/data1/akane/LLaVA/data \
    --multiple_vision_towers openai/clip-vit-large-patch14-336 facebook/dinov2-large \
    --pretrain_mm_mlp_adapter /data/data1/akane/pretrained/mm_projector_7b.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/data0/akane/multi-ve-shared-resampler-clip-dino-llava-pretrained-v1.5-7b/checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name multi-ve-shared-resampler-clip-dino-llava-pretrained-7b-it 

