
import argparse
from ast import arg
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
parser.add_argument('--model_name', type=str, default='llava-v1.5-7b-lora', help='Model name (default: llava-v1.5-7b-lora)')
parser.add_argument('--model_path', type=str, default='./', help='Model path (default: ./)')
parser.add_argument('--model_base', type=str, default=None, help='Model base (path to the base LLM, used only for lora) (default: None)')
parser.add_argument('--data_path', type=str, default='./smart_watch_test.json', help='path to the test data file (default: ./smart_watch_test.json)')
parser.add_argument('--image_folder', type=str, default='./smart_watch_image_test', help='path to the test image folder (default: ./smart_watch_image_test)')
parser.add_argument('--ground_truth_image_folder', type=str, default='./smart_watch_image_test', help='path to the ground truth (not generated) image folder (default: ./smart_watch_image_test)')
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
model_list=[f'{args_main.model_path}{model_name}/checkpoint-{i}' for i in ckp_list]
for k in range(len(model_list)):
    args = type('Args', (), {
        "model_path": model_list[k],
        "model_base": args_main.model_base,
        "data_path": args_main.data_path,
        "image_folder": args_main.image_folder,
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
       
    
    print(f"inference end, answer saved to {ans_file}")
    ans_file.close() 

# begin evaluation
# evaluation fo VQA

import json
from tqdm import tqdm
import re
import torch
import pandas as pd

model_name=args_main.model_name
answer_list=ckp_list

# 定义 color_mapping
color_mapping = {
    'red': [0.1, 0.15],
    'green': [0.15, 0.3],
    'blue': [0.3, 0.45],
    'yellow': [0.45, 0.6],
    'orange': [0.6, 0.75],
    'purple': [0.75, 0.9]
}

# 提取所有颜色
colors = color_mapping.keys()

# 检查颜色是否存在于字符串中
def find_colors_in_string(input_string):
    found_colors = [color for color in colors if color in input_string]
    return found_colors


def map_to_color(pixel):
    if pixel<0.1:
        return 'black'
    elif 0.1<=pixel<0.15:
        return 'red'
    elif 0.15<=pixel<0.3:
        return 'green'
    elif 0.3<=pixel<0.45:
        return 'blue'
    elif 0.45<=pixel<0.6:
        return 'yellow'
    elif 0.6<=pixel<0.75:
        return 'orange'
    elif 0.75<=pixel<=0.9:
        return 'purple'
    else:
        return 'other'

def compare_img(gen_img,gt_img):
    correct_pixel=0
    incorrect_pixel=0
    for i in range(len(gen_img)):
        for j in range(len(gen_img[i])):
            if map_to_color(gen_img[i][j])!=map_to_color(gt_img[i][j]):
                incorrect_pixel+=1
            else:
                correct_pixel+=1
    return correct_pixel,incorrect_pixel

acc=[]
for answer in answer_list:    
    data_path=f"./answer/answer-{model_name}-{answer}.jsonl"
    time_count=0
    time_score=0
    weather_count=0
    weather_score=0
    position_count=0
    position_score=0
    battery_count=0
    battery_score=0
    with open (data_path, "r") as f:
        for line in tqdm(f):
            json_obj = json.loads(line.strip())
            ground_truth=json_obj['groun_truth']
            answer=json_obj['answer']
            prompt=json_obj['prompt']
            if prompt[0]!='<':
                continue
            if ' ' not in ground_truth:
                if ':' in ground_truth:
                    time_count+=1
                    pattern = r"(\d{2}):(\d{2}):(\d{2})"
                    match = re.search(pattern, ground_truth)
                    gt_h = int(match.group(1))
                    gt_m = int(match.group(2))
                    gt_s = int(match.group(3))
                    match = re.search(pattern, answer)
                    if match:
                        ans_h = int(match.group(1))
                        ans_m = int(match.group(2))
                        ans_s = int(match.group(3))
                        err_h=abs(ans_h - gt_h)
                        err_h=min(err_h,12-err_h)
                        err_m=abs(ans_m - gt_m)
                        err_m=min(err_m,60-err_m)
                        err_s=abs(ans_s - gt_s)
                        err_s=min(err_s,60-err_s)
                        err=(err_h/6.0 + err_m/30.0 + err_s/30.0)/3.0
    
                        time_score+=1-err
                elif 'sunny' in ground_truth or 'raining' in ground_truth or 'cloudy' in ground_truth:
                    weather_count+=1
                    if ground_truth in answer:
                        weather_score+=1
                elif '-' in ground_truth:
                    position_count+=1
                    if ground_truth in answer:
                        position_score+=1
                elif '%' in ground_truth:
                    battery_count+=1
                    match = re.search(r'\b(100|[1-9]\d?|0)%', ground_truth)
                    gt=int(match.group(1)) / 100
                    match = re.search(r'\b(100|[1-9]\d?|0)%', answer)
                    if match:
                        ans=int(match.group(1)) / 100
                        err=abs(ans - gt)
                        battery_score+=1-err
                else:
                    raise ValueError(f"Unknown ground truth format: {ground_truth}")
    time_acc=time_score/time_count
    weather_acc=weather_score/weather_count
    position_acc=position_score/position_count
    battery_acc=battery_score/battery_count
    total_acc=(time_score+weather_score+position_score+battery_score)/(time_count+weather_count+position_count+battery_count)
    acc.append([time_acc,weather_acc,position_acc,battery_acc,total_acc])
acc=torch.Tensor(acc)
acc=acc.permute(1,0)
columns = [f"{i*10}%" for i in range(1, 1+acc.shape[1])]
row_index = ['time_acc', 'weather_acc', 'position_acc', 'battery_acc', 'total_acc']
df = pd.DataFrame(data=acc.numpy(),
                  index=row_index,
                  columns=columns)
df.to_csv("VQA_results.csv", index=True, header=True)


#begin image generation evaluation

import os
from PIL import Image
import torch
from tqdm import tqdm


f"./answer/answer-{model_name}-{ckp_list[k]}-image"

# 原始路径列表
source_paths = [
    f"./answer/answer-{model_name}-{i*args_main.ckpt_step}-image"
    for i in range(args_main.ckpt_start,args_main.ckpt_num+args_main.ckpt_start)
]

print('transfering images from .pt to .png')

for source_path in tqdm(source_paths):
    # 创建目标目录（原目录名 + "-transferred"）
    target_dir = f"{source_path}-transferred"
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录中的所有.pt文件
    for pt_file in os.listdir(source_path):
        if not pt_file.endswith('.pt'):
            continue
        
        # 构造完整的文件路径
        pt_path = os.path.join(source_path, pt_file)
        png_filename = os.path.splitext(pt_file)[0] + '.png'
        png_path = os.path.join(target_dir, png_filename)
        
        try:
            # 加载.pt文件（假设存储的是图像张量）
            img_tensor = torch.load(pt_path)
            
            # 转换为PIL图像
            img = Image.fromarray(img_tensor.numpy())
            
            # 保存为PNG
            img.save(png_path)
            #print(f"Converted: {pt_path} -> {png_path}")
            
        except Exception as e:
            print(f"Error converting {pt_path}: {str(e)}")

import subprocess
import csv


def calculate_fid(converted_path):
    """调用pytorch_fid计算FID分数"""
    try:
        cmd = [
            "python", "-m", "pytorch_fid",
            converted_path,
            GT_PATH,
            "--device", CUDA_DEVICE
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 解析输出结果
        for line in result.stdout.split('\n'):
            if line.startswith("FID:"):
                return float(line.split()[1])
                
    except subprocess.CalledProcessError as e:
        print(f"FID计算失败: {e.stderr}")
        return "ERROR"
    except Exception as e:
        print(f"解析错误: {str(e)}")
        return "PARSE_ERROR"
    return None

GT_PATH = args_main.ground_truth_image_folder
CUDA_DEVICE = args_main.device
RESULT_CSV = "./fid_results.csv"

# 初始化结果存储
fid_results = []

# 计算所有FID
for source_path in source_paths:
    target_dir = f"{source_path}-transferred"
    print(f"Calculating FID for {target_dir}")
    fid_score = calculate_fid(target_dir)
    fid_results.append(fid_score)
    print(f"FID计算完成: {fid_score}")

# 写入CSV文件（单行）
with open(RESULT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # 创建表头
    headers = [f"Model_{i+1}" for i in range(len(source_paths))]
    
    # 写入表头和结果
    writer.writerow(headers)
    writer.writerow(fid_results)