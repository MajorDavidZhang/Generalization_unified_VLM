#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1,2,3
#--generation_only True \
#deepspeed --master_port 29501 --include=localhost:2,3 /datadrive_a/jihai/LLaVA/llava/train/train.py \
#siglip have no cls token, mm_vision_select_feature should be set to cls_patch
#bs 85 if 3 gpu
#--vision_tower google/siglip-base-patch16-224 \
    # --vision_tower_gen synthetic \
    # --vision_tower_gen_path /datadrive_a/jihai/LLaVA/llava/model/multimodal_encoder/plain.pth\
    # --vision_tower_permutation_path /public_data/jihai/understanding/llava/model/multimodal_encoder/siglip_affine_1.pth\
# 禁用RoCE，强制使用本地总线
export HF_ENDPOINT=https://hf-mirror.com
deepspeed --master_port 29506 --include=localhost:0,1,2,3,4,5,6,7 /public_data/jihai/understanding/llava/train/train_mem.py \
    --deepspeed /public_data/jihai/understanding/scripts/zero2.json \
    --model_name_or_path /public_data/jihai/tmp/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /public_data/ShareGPT4V/sharegpt4v/blip_laion_cc_sbu_558k_with_generation.json \
    --image_folder /public_data/ShareGPT4V/llava/llava_pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_path /public_data/jihai/tmp/clip-336 \
    --detach_mm_projector True \
    --mm_projector_type mlp \
    --tune_mm_mlp_adapter True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature patch \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --understanding_only False \
    --dataset llava \
    --image_loss cosine \
    --image_shape_un 3 336 336 \
    --image_shape_gen 3 336 336 \
    --num_image_token 576 \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/llava-v1.5-7b-clip-gen-detach-pretrain \
    --num_ckpt_to_save 1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

