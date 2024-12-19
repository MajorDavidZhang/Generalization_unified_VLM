#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=localhost --master_port=26105 /datadrive_a/jihai/LLaVA/llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path /datadrive_a/jihai/azure_storage2/vigstandard_data/jihai/checkpoint/vicuna-7b-v1.5/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /datadrive_a/jihai/data/multimodalout/dummy_data.json \
    --image_folder /datadrive_a/jihai/LLaVA/dummy_data/images \
    --vision_tower synthetic \
    --pretrain_mm_mlp_adapter /datadrive_a/jihai/LLaVA/scripts/v1_5/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
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
