# to inject affine transformation, using "--vision_tower_permutation_path ./llava/model/multimodal_encoder/vqaffine_0.1.pth \"
deepspeed --master_port 29516 --include=localhost:5,6,7 ./llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /public_data/jihai/tmp/vicuna-7b-v1.5 \
    --version v1 \
    --data_path .../smart_watch_train.json \
    --image_folder .../smart_watch_image_train \
    --vision_tower vq \
    --vision_tower_gen vq \
    --mm_projector_head_output_size 16384 \
    --mm_projector_type linear \
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
    --image_shape_un 3 256 256 \
    --image_shape_gen 3 256 256 \
    --num_image_token 256 \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/llava-v1.5-7b-vq-vq-lora \
    --num_ckpt_to_save 10 \
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

# evaluate ckpt 10 only
python eval_generate.py \
  --device "cuda:7" \
  --ckpt_start 10 \
  --ckpt_step 45 \
  --ckpt_num 1 \
  --model_name "llava-v1.5-7b-vq-vq-lora" \
