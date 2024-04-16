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
    --resampler_grid_size 24\
    --use_brave_adapters True \
    --pretrain_mm_mlp_adapter /data/data1/akane/mve-clip-dino-brave-pretrain/checkpoints/mm_projector.bin \
    --pretrain_resampler /data/data1/akane/mve-clip-dino-brave-pretrain/checkpoints/resampler.bin \
    --scaled_clip_residual True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/data1/akane/mve-clip-dino-brave-finetune/checkpoints/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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
    --run_name mve-clip-dino-brave-finetune-7b