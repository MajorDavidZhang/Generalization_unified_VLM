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
deepspeed --master_port 29515 --include=localhost:5,6,7 /public_data/jihai/understanding/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4 \
    --deepspeed /public_data/jihai/understanding/scripts/zero2.json \
    --model_name_or_path /public_data/jihai/tmp/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /public_data/jihai/data/multimodalout/smart_watch_train_68kg_180km.json \
    --image_folder /public_data/jihai/data/multimodalout/smart_watch_image_train_68kg_180km \
    --vision_tower google/siglip-base-patch16-224 \
    --vision_tower_path /public_data/jihai/tmp/siglip-base-patch16-224\
    --vision_tower_gen vq \
    --mm_projector_head_output_size 16384 \
    --mm_projector_type mlp \
    --mm_projector_gen_type linear \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature cls_patch \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --understanding_only False \
    --dataset smartwatch \
    --image_loss cosine \
    --alpha 0.2 \
    --image_shape_un 3 224 224 \
    --image_shape_gen 3 256 256 \
    --num_image_token 256 \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/llava-v1.5-7b-siglip-vq_68kg_180km-sw-lora \
    --num_ckpt_to_save 14 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 131 \
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
    --report_to none

sleep 10

python eval_generate_smartwatch.py \
  --device "cuda:5" \
  --ckpt_start 5 \
  --ckpt_step 45 \
  --ckpt_num 3 \
  --model_name "llava-v1.5-7b-siglip-vq_68kg_180km-sw-lora" > output_gpu5.log 2>&1 &

python eval_generate_smartwatch.py \
  --device "cuda:6" \
  --ckpt_start 8 \
  --ckpt_step 45 \
  --ckpt_num 3 \
  --model_name "llava-v1.5-7b-siglip-vq_68kg_180km-sw-lora" > output_gpu6.log 2>&1 &

python eval_generate_smartwatch.py \
  --device "cuda:7" \
  --ckpt_start 11 \
  --ckpt_step 45 \
  --ckpt_num 4 \
  --model_name "llava-v1.5-7b-siglip-vq_68kg_180km-sw-lora" \
