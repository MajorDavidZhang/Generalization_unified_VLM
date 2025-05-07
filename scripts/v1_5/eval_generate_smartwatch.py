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
import torch.nn.functional as F
from PIL import Image
import math


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self,args, tokenizer, model_config,generation_only=False, understanding_only=False):
        self.list_data_dict = json.load(open(args.data_path, "r"))
        if generation_only:
            self.list_data_dict = [e for e in self.list_data_dict if e['task']=="generation"]
        if understanding_only:
            self.list_data_dict = [e for e in self.list_data_dict if (e['task']=="vqa" or e['task']=="caption")]

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.image_folder = args.image_folder
        self.gen_processor=args.gen_processor
        self.un_processor=args.un_processor
        self.conv_mode = args.conv_mode

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        
        qs = sources["conversations"][0]["value"]
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print(prompt)
        image_file = sources["image"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_un=None
        image_gen=None
        if sources['task']=='generation':
            image_gen = self.gen_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image_un = self.un_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_un,image_gen

    def __len__(self):
        return len(self.list_data_dict)


def collate_fn(batch):
    input_ids, image_un,image_gen = zip(*batch)
    image_un= [img for img in image_un if img is not None]
    image_gen= [img for img in image_gen if img is not None]
    input_ids = torch.stack(input_ids, dim=0)
    image_un = torch.stack(image_un, dim=0) if len(image_un) > 0 else None
    image_gen = torch.stack(image_gen, dim=0) if len(image_gen) > 0 else None
    images={'images_un':image_un,'images_gen':image_gen}
    return input_ids, images


# DataLoader
def create_data_loader(args, tokenizer, model_config, batch_size=1, num_workers=4,understanding_only=False,generation_only=False):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(args, tokenizer, model_config,understanding_only=understanding_only,generation_only=generation_only)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader,dataset

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


def generate_image_vq(input_ids,model,num_image_tokens):
    output_img_id=[]
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
            img_logits = model.get_model().mm_projector_head(hidden_states[:,-1,:])
            img_id=img_logits.argmax(dim=-1) # shape (1,)
            output_img_id.append(img_id)
            img_latent=model.get_model().vision_tower_gen.vision_tower.quantize.get_codebook_entry(img_id, shape=None, channel_first=True) # (1,8)
            if model.get_model().mm_projector_gen is not None:
                new_embed=model.get_model().mm_projector_gen(img_latent)
            else:
                new_embed=model.get_model().mm_projector_un(img_latent)
            new_embed=new_embed.unsqueeze(1).to(inputs_embeds.device)
            inputs_embeds=torch.cat([inputs_embeds,new_embed],dim=1)
            
    return output_img_id




# 解析命令行参数
parser = argparse.ArgumentParser(description="Run model evaluation with configurable parameters.")
parser.add_argument('--device', type=str, default='cuda:7', help='Device to use (default: cuda:7)')
parser.add_argument('--ckpt_start', type=int, default=1, help='Start multiplier for checkpoints (default: 1)')
parser.add_argument('--ckpt_step', type=int, default=30, help='Step multiplier for checkpoints (default: 30)')
parser.add_argument('--ckpt_num', type=int, default=10, help='Number of checkpoints (default: 10)')
parser.add_argument('--model_name', type=str, default='llava-v1.5-7b-sw-u-lora', help='Model name (default: llava-v1.5-7b-sw-u-lora)')
parser.add_argument('--understanding_only', action='store_true', default=False, help='Enable understanding only mode (default: True)')
parser.add_argument('--generation_only', action='store_true', default=False, help='Enable generation only mode (default: False)')
parser.add_argument('--generate_mode', type=str, default='vq')
args_main = parser.parse_args()

#load trained model
device=args_main.device
ckp_list=[i*args_main.ckpt_step for i in range(args_main.ckpt_start,args_main.ckpt_num+args_main.ckpt_start)]
model_name=args_main.model_name
understanding_only=args_main.understanding_only
generation_only=args_main.generation_only
model_list=[f'/public_data/jihai/understanding/scripts/v1_5/checkpoints/{model_name}/checkpoint-{i}' for i in ckp_list]
for k in range(len(model_list)):
    args = type('Args', (), {
        "model_path": model_list[k],
        "model_base": '/public_data/jihai/tmp/vicuna-7b-v1.5',
        "data_path": '/public_data/jihai/data/multimodalout/smart_watch_test.json',
        "image_folder": '/public_data/jihai/data/multimodalout/smart_watch_image_test',
        "answers_file": f"./answer/answer-{model_name}-{ckp_list[k]}.jsonl",
        "answer_image_file": f"./answer/answer-{model_name}-{ckp_list[k]}-image",
        "conv_mode": "llava_v1",
        "num_chunks": 1,
        "chunk_idx": 0,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 128,
        "image_un_size": [3,224,224],
        "image_gen_size": [3,256,256]
    })()
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_type = get_model_name_from_path(model_path)
    tokenizer, model, image_processor,image_processor_gen, context_len = load_pretrained_model(model_path, args.model_base, model_name,device=device)
    args.gen_processor=image_processor_gen
    args.un_processor=image_processor


    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(args.answer_image_file, exist_ok=True)
    ans_file = open(answers_file, "w")
    if 'plain' in model_type and 'finetune' not in model_type.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader,data_set = create_data_loader(args, tokenizer, model.config,understanding_only=understanding_only,generation_only=generation_only)
    list_data_dict = data_set.list_data_dict

    images_gen_pad=torch.zeros([0]+args.image_gen_size).to(device=device, dtype=torch.float16)
    images_un_pad=torch.zeros([0]+args.image_un_size).to(device=device, dtype=torch.float16)
    count=0
    for (input_ids, images), line in tqdm(zip(data_loader, list_data_dict), total=len(list_data_dict)):
        count+=1
        if count==500: break
    
        cur_prompt = line["conversations"][0]["value"]
        groun_truth=line["conversations"][1]["value"]
        groun_truth_img_tensor=line["image"]
        input_ids = input_ids.to(device=device, non_blocking=True)
        images['images_gen']=images['images_gen'].to(dtype=torch.float16, device=device, non_blocking=True) if images['images_gen'] is not None else images_gen_pad
        images['images_un']=images['images_un'].to(dtype=torch.float16, device=device, non_blocking=True) if images['images_un'] is not None else images_un_pad
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=images,
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
        img_file=None
        if start_idx != -1:
            output_ids=output_ids[:,1:start_idx+3]
            input_ids=torch.cat((input_ids, output_ids), dim=1)
            if args_main.generate_mode=='vq':
                img_id=generate_image_vq(input_ids,model,model.get_model().vision_tower_gen.num_patches)
                with torch.no_grad():
                    img=model.get_model().vision_tower_gen.vision_tower.decode_code(img_id,[1,8,16,16])
                img = F.interpolate(img, size=[args.image_gen_size[1], args.image_gen_size[2]], mode='bicubic').permute(0, 2, 3, 1)[0]
                img = torch.clamp(127.5 * img + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
                img_file=os.path.join(args.answer_image_file, f'{count}.pt')
                torch.save(img, img_file)
            else:
                img=generate_image(input_ids,model,model.get_model().vision_tower_gen.num_patches)
                img=torch.stack(img,dim=0).squeeze().cpu()
                img_file=os.path.join(args.answer_image_file, f'{count}.pt')
                torch.save(img, img_file)

        ans_file.write(json.dumps({"prompt": cur_prompt,
                                    "groun_truth": groun_truth,
                                    "answer": outputs,
                                    "groun_truth_img_tensor": groun_truth_img_tensor,
                                    "output_img_file": img_file,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
        #outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0].strip()
    
    print(ans_file)
    ans_file.close() 
