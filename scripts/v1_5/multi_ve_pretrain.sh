#!/bin/bash

echo "\n================================================= $(date -u) =========================================================\n"
export WANDB_PROJECT="multi-ve-llava"
deepspeed llava/train/multi_ve_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data/data1/akane/LLaVA/data/blip_laion_cc_sbu_558k.json \
    --image_folder /data/data1/akane/LLaVA/data/blip_laion_558k \
    --multiple_vision_towers openai/clip-vit-large-patch14-336 facebook/dinov2-large \
    --use_brave_adapters True \
    --resampler_grid_size 24 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_mm_resampler True \
    --scaled_clip_residual False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/data1/akane/mve-clip-dino-brave-pretrain/checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --run_name mve-clip-dino-brave-pretrain-7b
