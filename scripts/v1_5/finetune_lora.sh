#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1,2,3
#
#deepspeed --master_port 29501 --include=localhost:2,3 /datadrive_a/jihai/LLaVA/llava/train/train.py \
#siglip have no cls token, mm_vision_select_feature should be set to cls_patch
#bs 85 if 3 gpu
    # --vision_tower_gen synthetic \
    # --vision_tower_gen_path /datadrive_a/jihai/LLaVA/llava/model/multimodal_encoder/plain.pth\
deepspeed --master_port 29505 --include=localhost:1,2 /datadrive_a/jihai/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-3 \
    --deepspeed /datadrive_a/jihai/LLaVA/scripts/zero2.json \
    --model_name_or_path /datadrive_a/tmp/vicuna-7b-v1.5/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /datadrive_a/jihai/data/multimodalout/dummy_data.json \
    --image_folder /datadrive_a/jihai/data/multimodalout/smart_watch_image_train \
    --vision_tower synthetic \
    --vision_tower_path /datadrive_a/jihai/LLaVA/llava/model/multimodal_encoder/plain.pth\
    --mm_projector_type linear \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature cls_patch \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --understanding_only True \
    --dataset segment_digit \
    --image_loss mse \
    --image_shape 6 7 \
    --num_image_token 6 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-segment-digit-lora \
    --num_ckpt_to_save 5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
