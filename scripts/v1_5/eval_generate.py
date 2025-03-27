import argparse
from email.mime import image
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset_segment_digit(Dataset):
    def __init__(self,args, tokenizer, model_config):
        self.list_data_dict = json.load(open(args.data_path, "r"))
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.image_folder = args.image_folder
        self.gen_processor=args.gen_processor
        self.un_processor=args.un_processor

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        qs = sources["conversations"][0]["value"]
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print(prompt)
        #image_file = sources["image"]
        #image_tensor = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor=torch.Tensor(sources['tensor'])
        # if sources['task']=='generation':
        #     image_tensor = self.gen_processor.preprocess(image_tensor, return_tensors='pt')['pixel_values'][0]
        # else:
        #     image_tensor = self.un_processor.preprocess(image_tensor, return_tensors='pt')['pixel_values'][0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.list_data_dict)


def collate_fn(batch):
    input_ids, image_tensors = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors


# DataLoader
def create_data_loader(args, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset_segment_digit(args, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def generate_image(input_ids,model,num_image_tokens):
    output_img=[]
    inputs_embeds=model.get_model().embed_tokens(input_ids) #1, seq_le, 4096
    with torch.inference_mode():
        for i in range(num_image_tokens):
            outputs = model.model(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = outputs[0]
            img = model.get_model().mm_projector_head(hidden_states[:,-1,:])
            output_img.append(img)
            if model.get_model().mm_projector_gen is not None:
                new_embed=model.get_model().mm_projector_gen(img)
            else:
                new_embed=model.get_model().mm_projector_un(img)
            new_embed=new_embed.unsqueeze(1).to(inputs_embeds.device)
            inputs_embeds=torch.cat([inputs_embeds,new_embed],dim=1)
            
    return output_img

#load trained model
device='cuda:1'
ckp_list=[i*78 for i in range(1,6)]
model_name='llava-v1.5-7b-segment-digit-lora'
model_list=[f'/datadrive_a/jihai/LLaVA/scripts/v1_5/checkpoints/{model_name}/checkpoint-{i}' for i in ckp_list]
for k in range(len(model_list)):
    args = type('Args', (), {
        "model_path": model_list[k],
        "model_base": '/datadrive_a/tmp/vicuna-7b-v1.5/vicuna-7b-v1.5',
        "data_path": '/datadrive_a/jihai/data/multimodalout/dummy_data_eval.json',
        "image_folder": '/datadrive_a/jihai/data/multimodalout/smart_watch_image_test',
        "answers_file": f"./answer/answer-{model_name}-{ckp_list[k]}.jsonl",
        "conv_mode": "llava_v1",
        "num_chunks": 1,
        "chunk_idx": 0,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 128
    })()
    print(args.model_path)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print(model_path)
    model_type = get_model_name_from_path(model_path)
    tokenizer, model, image_processor,image_processor_gen, context_len = load_pretrained_model(model_path, args.model_base, model_name,device=device)
    args.gen_processor=image_processor_gen
    args.un_processor=image_processor


    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    if 'plain' in model_type and 'finetune' not in model_type.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(args, tokenizer, model.config)
    list_data_dict = json.load(open(args.data_path, "r"))


    count=0
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, list_data_dict), total=len(list_data_dict)):
        count+=1
        if count==500: break
    
        cur_prompt = line["conversations"][0]["value"]
        groun_truth=line["conversations"][1]["value"]
        groun_truth_img_tensor=line["tensor"]
        input_ids = input_ids.to(device=device, non_blocking=True)
        image_tensor=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        output_ids=outputs['generated_tokens']
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        #print(outputs)

        img_indicator = torch.tensor([529,  3027, 29958])
        id_seq = output_ids[0].cpu()

        # 子序列长度
        sub_seq_len = len(img_indicator)

        # 滑动窗口查找子序列
        start_idx = -1
        for i in range(id_seq.size(0) - sub_seq_len + 1):
            if torch.equal(id_seq[i:i + sub_seq_len], img_indicator):
                start_idx = i
                break
        img=None
        if start_idx != -1:
            output_ids=output_ids[:,1:start_idx+3]
            input_ids=torch.cat((input_ids, output_ids), dim=1)
            img=generate_image(input_ids,model,model.get_model().vision_tower.num_patches)
            img=torch.stack(img,dim=0).squeeze().cpu().tolist()
        ans_file.write(json.dumps({"prompt": cur_prompt,
                                    "groun_truth": groun_truth,
                                    "answer": outputs,
                                    "groun_truth_img_tensor": groun_truth_img_tensor,
                                    "output_img_tensor": img,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
        #outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0].strip()
    
    print(ans_file)
    ans_file.close() 
