# Are Unified Vision-Language Models Necessary: Generalization Across Understanding and Generation




## Install


1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install -e .
```
Please also refer to ```requirements.txt```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

3. Download ```vq_ds16_c2i.pt``` to ```./llava/model/multimodal_encoder/``` from [llamaGen](https://huggingface.co/FoundationVision/LlamaGen/tree/main)

4. Install [pytorch-fid](https://github.com/mseitzer/pytorch-fid) to do evaluation on image generation



### Experiments on Synthetic Dataset

1. Generate the training and testing dataset using ```./generate_data_smart_watch.ipynb```

2. Use the ```finetune_lora*.sh``` in ```./scripts/v1_5/``` to do training and evaluation. To test the model with affine-transformation, use ```./llava/model/multimodal_encoder/affine_transformation_generation.ipynb``` to generate the transformation first and then change the finetune_lora bash file accordingly.

### Experiments on LLaVA-1.5 Dataset

1. Prepare the ShareGPT4V dataset following [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V). You do not need to download the images from SAM, since we will bypass them in preprocessing to save time. Images in ShareGPT4V dataset include images in LLaVA-1.5 dataset. Also download the text part of LLaVA-1.5 dataset from [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)

2. Use ```./scripts/v1_5/caption-to-image-generation.ipynb``` to transform the ShareGPT4V image caption data into text-to-image generation data, and append to the original LLaVA-1.5 dataset.

3. Use the ```pretrain*.sh``` and ```finetune*.sh``` in ```./scripts/v1_5/``` to do the two-stage training. You can refer to [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) for more details.

4. Install the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to do evaluation. Please replace the ```lmms-eval/lmms_eval/models/llava.py``` in the lmms-eval code with ```./lmms-eval/llava.py```. Example command:
```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="./scripts/v1_5/checkpoints/llava-v1.5-7b/checkpoint-6761" \
    --tasks pope,textvqa,mmvet,vizwiz_vqa,gqa,mmbench_en,mme,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5-7b \
    --output_path ./logs/
```

### Bug Fixing

A few matters need attention:
- Please manually add `do_sample:true` in in vicuna's generation_config.json file, according to this [issue](https://github.com/haotian-liu/LLaVA/issues/1144)
- We use `zero2` settings in visual instruction tuning, because `zero3` may cause some unknow timeout error. Please set `"overlap_comm": false` to avoid zero loss error, according to this [issue](https://github.com/haotian-liu/LLaVA/issues/1231)


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.


