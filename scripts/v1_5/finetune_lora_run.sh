# bash finetune_lora_vq-vq-3.sh
# bash finetune_lora_vq-vq-4.sh
#vq-vq-u vq-vq vq-vq-detach siglip-siglip-detach siglip-u siglip-siglip 
# bash finetune_lora_vq-vq.sh
# bash finetune_lora_vq-vq-2.sh
# bash finetune_lora_vq-vq-3.sh
# bash finetune_lora_siglip-siglip.sh
# bash finetune_lora_siglip-siglip-2.sh
# bash finetune_lora_siglip-siglip-3.sh
# bash finetune_lora_siglip-vq-3.sh
# bash finetune_lora_vq-siglip.sh
# bash finetune_lora_vq-siglip-2.sh
bash pretrain.sh
sleep 10
bash finetune.sh