#!/bin/bash

export WANDB_PROJECT="grllava"
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path /data/data1/akane/LLaVA/data/llava_v1_5_mix665k.json \
    --image_folder /data/data1/akane/LLaVA/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/data1/akane/pretrained/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_use_additional_adapter False \
    --mm_vision_use_pretrained_additional_adapter False \
    --mm_vision_use_global_tokens False \
    --mm_vision_use_granular_tokens False \
    --mm_vision_use_scaled_residual_granular_tokens False \
    --mm_vision_use_static_scaled_residual_granular_tokens True \
    --mm_vision_use_static_residual_scaler True \
    --mm_vision_use_residual_scaler False \
    --mm_vision_num_tokens_per_layer 576 \
    --mm_vision_granular_select_layers "6 12 18" \
    --mm_vision_granular_tokens_strategy "pool" \
    --mm_vision_granular_tokens_per_layer 192 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/data0/akane/static-residual-grllava-pretrained-v1.5-7b/checkpoints/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
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
    --run_name static-residual-grllava-pretrained-7b-it 




# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb
