#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from email.mime import image
import os
from pyexpat import model

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower#,build_vision_tower_inverse
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class VisionArguments:
    vision_tower: Optional[str] = field(default=None)
    vision_tower_path: Optional[str] = field(default=None)
    vision_tower_permutation_path: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector_un=build_vision_projector(config.mm_un_hidden_size,config.hidden_size,self.config.mm_projector_type)
            self.config.detach_mm_projector = getattr(config, 'detach_mm_projector', False)
            self.config.mm_projector_gen_type = getattr(config, 'mm_projector_gen_type', self.config.mm_projector_type)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
        if hasattr(config, "mm_vision_tower_gen"):
            if config.mm_vision_tower_gen == 'same':
                self.vision_tower_gen = self.vision_tower
                if self.config.detach_mm_projector:
                    self.mm_projector_gen = build_vision_projector(config.mm_gen_hidden_size,config.hidden_size,self.config.mm_projector_gen_type)
                else:
                    self.mm_projector_gen = None
            else:
                vision_tower_gen_cfg=VisionArguments(
                    vision_tower=config.mm_vision_tower_gen,
                    vision_tower_path=config.vision_tower_gen_path,
                    mm_vision_select_layer=config.mm_vision_select_layer,
                    mm_use_im_start_end=config.mm_use_im_start_end,
                    mm_use_im_patch_token=config.mm_use_im_patch_token,
                    mm_patch_merge_type=config.mm_patch_merge_type,
                    mm_vision_select_feature=config.mm_vision_select_feature,
                    vision_tower_permutation_path=config.vision_tower_gen_permutation_path
                )
                self.vision_tower_gen = build_vision_tower(vision_tower_gen_cfg, delay_load=True)
                #self.config.mm_projector_gen_type = getattr(config, 'mm_projector_gen_type', self.config.mm_projector_type)
                
                self.mm_projector_gen = build_vision_projector(config.mm_gen_hidden_size,config.hidden_size,self.config.mm_projector_gen_type)
            self.mm_projector_head=build_vision_projector(config.hidden_size,config.mm_projector_head_output_size,'linear')

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_gen(self):
        vision_tower_gen = getattr(self, 'vision_tower_gen', None)
        if type(vision_tower_gen) is list:
            vision_tower_gen = vision_tower_gen[0]
        return vision_tower_gen
    

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_tower_gen= model_args.vision_tower_gen
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrained_mm_mlp_adapter = model_args.pretrained_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_gen = vision_tower_gen
        self.config.vision_tower_permutation_path=model_args.vision_tower_permutation_path
        self.config.vision_tower_gen_permutation_path=model_args.vision_tower_gen_permutation_path
        self.config.image_loss=model_args.image_loss
        self.config.alpha=model_args.alpha

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if model_args.vision_tower_gen=='same':
            self.vision_tower_gen = vision_tower

        else:
            if self.get_vision_tower_gen() is None:
                vision_tower_gen_cfg=VisionArguments(
                    vision_tower=model_args.vision_tower_gen,
                    vision_tower_path=model_args.vision_tower_gen_path,
                    mm_vision_select_layer=model_args.mm_vision_select_layer,
                    mm_use_im_start_end=model_args.mm_use_im_start_end,
                    mm_use_im_patch_token=model_args.mm_use_im_patch_token,
                    mm_patch_merge_type=model_args.mm_patch_merge_type,
                    mm_vision_select_feature=model_args.mm_vision_select_feature,
                    vision_tower_permutation_path=model_args.vision_tower_gen_permutation_path
                )
                vision_tower_gen = build_vision_tower(vision_tower_gen_cfg)
                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower_gen = [vision_tower_gen]
                else:
                    self.vision_tower_gen = vision_tower_gen
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower_gen = self.vision_tower_gen[0]
                else:
                    vision_tower_gen = self.vision_tower_gen
                vision_tower_gen.load_model()


        self.config.use_mm_proj = True
        self.config.vision_tower_path=model_args.vision_tower_path
        self.config.vision_tower_gen_path=model_args.vision_tower_gen_path
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_projector_gen_type = getattr(model_args, 'mm_projector_gen_type',self.config.mm_projector_type)
        self.config.detach_mm_projector = getattr(model_args, 'detach_mm_projector', False)
        self.config.mm_un_hidden_size = vision_tower.hidden_size
        self.config.mm_gen_hidden_size=self.vision_tower_gen.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        if model_args.mm_projector_head_output_size is not None:
            self.config.mm_projector_head_output_size=model_args.mm_projector_head_output_size
        else:
            self.config.mm_projector_head_output_size=self.config.mm_gen_hidden_size

        if getattr(self, 'mm_projector_un', None) is None:
            self.mm_projector_un=build_vision_projector(self.config.mm_un_hidden_size,self.config.hidden_size,self.config.mm_projector_type)
            if model_args.vision_tower_gen=='same':
                if self.config.detach_mm_projector:
                    self.mm_projector_gen = build_vision_projector(self.config.mm_gen_hidden_size,self.config.hidden_size,self.config.mm_projector_gen_type)
                else:
                    self.mm_projector_gen = None
            else:
                self.mm_projector_gen = build_vision_projector(self.config.mm_gen_hidden_size,self.config.hidden_size,self.config.mm_projector_gen_type)
            self.mm_projector_head=build_vision_projector(self.config.hidden_size,self.config.mm_projector_head_output_size,'linear')

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector_un.parameters():
                p.requires_grad = True
            if self.mm_projector_gen is not None:
                for p in self.mm_projector_gen.parameters():
                    p.requires_grad = True
            for p in self.mm_projector_head.parameters():
                p.requires_grad = True

        if pretrained_mm_mlp_adapter is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            #mm_projector_un_weights_path=os.path.join(pretrained_mm_mlp_adapter,'mm_projector_un.pth')
            mm_projector_weights = torch.load(pretrained_mm_mlp_adapter, map_location='cpu')
            self.mm_projector_un.load_state_dict(get_w(mm_projector_weights, 'mm_projector_un'))

            #mm_projector_head_weights_path=os.path.join(pretrained_mm_mlp_adapter,'mm_projector_head.pth')
            #mm_projector_head_weights = torch.load(mm_projector_head_weights_path, map_location='cpu')
            self.mm_projector_head.load_state_dict(get_w(mm_projector_weights, 'mm_projector_head'))

            if self.mm_projector_gen is not None :
                #self.mm_projector_gen_weights_path=os.path.join(pretrained_mm_mlp_adapter,'mm_projector_gen.pth')
                #mm_projector_gen_weights = torch.load(self.mm_projector_gen_weights_path, map_location='cpu')
                self.mm_projector_gen.load_state_dict(get_w(mm_projector_weights, 'mm_projector_gen'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower_gen(self):
        return self.get_model().get_vision_tower_gen()

    # def encode_images(self, images):
    #     image_features = self.get_model().get_vision_tower()(images)
    #     image_features = self.get_model().mm_projector(image_features)
    #     return image_features



    def my_prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images,img_start_token, image_sizes=None, input_img_features=None
    ):
        #generate: position_ids, attention_mask, past_key_values, labels=None
        #input_img_features: override the image features, used for generation evaluation, shape (6,4096)

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels,None


        if type(images) is list :
            raise Exception("type(images) is list  not implemented")
        #else:
            #image_features = self.encode_images(images).to(self.device)
            #print(image_features.shape) #torch.Size([batch_size,seqlen, 4096])

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        image_labels=None
        images_generation = images['images_gen']  # 使用布尔索引选出对应的 images
        image_features_generation=None
        if images_generation.shape[0]>0:
            vision_tower_gen_out = self.get_model().get_vision_tower_gen()(images_generation)
            if isinstance(vision_tower_gen_out, tuple): #VQ
                image_features_generation = vision_tower_gen_out[0]
                image_labels=vision_tower_gen_out[1]
                image_labels=image_labels.view(image_features_generation.shape[0],-1) #b,seq_len
                image_features_generation = image_features_generation.to(self.device)
            else:
                image_features_generation = vision_tower_gen_out.to(self.device)
                image_labels=image_features_generation.detach().clone() #-------todo for VQ might need discretization
      
            
            # image_labels=image_labels[mask_not_none]
            if self.get_model().mm_projector_gen is not None:
                image_features_generation=self.get_model().mm_projector_gen(image_features_generation)
            else:
                image_features_generation=self.get_model().mm_projector_un(image_features_generation)
            #image_features_generation=image_features_generation[mask_not_none]
        
      
        # 找到 img_start_token 为 None 的部分
        images_understanding = images['images_un']  # 使用布尔索引选出对应的 images
        image_features_understanding=None
        if images_understanding.shape[0]>0:
            image_features_understanding = self.get_model().get_vision_tower()(images_understanding) #include pseudo padding image features,which should be skipped
            if isinstance(image_features_understanding, tuple):
                image_features_understanding = image_features_understanding[0]
            image_features_understanding = image_features_understanding.to(self.device)
            image_features_understanding=self.get_model().mm_projector_un(image_features_understanding)
        new_input_embeds = []
        new_labels = []
        cur_image_understanding_idx = 0
        cur_image_generation_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_input_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_generated_images = (cur_input_ids == -300).sum() #for generation process, where image marker is -300 not IMAGE_TOKEN_INDEX
            
            if num_input_images != 0 and num_generated_images == 0: #for understanding only process
            
                
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1): #remove image_token (-200)
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_input_images + 1): #one round may have multiple images
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_input_images:
                        cur_image_features = image_features_understanding[cur_image_understanding_idx] # 6,4096
                        if input_img_features is not None:
                            cur_image_features = input_img_features
                        #print(f"cur_image_features_shape{cur_image_features.shape}")
                        cur_image_understanding_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
            elif num_input_images == 0 and num_generated_images == 0: #text only process
               
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)   
                #cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_understanding_idx += 1 #skip the pseudo padding image features in data module (torch.zeros)
            elif num_input_images == 0 and num_generated_images != 0: #for generation only process
               
                cur_image_features = image_features_generation[cur_image_generation_idx]
                cur_input_ids[cur_input_ids == -300] = 0 #replace the image token with 0 so that embedding can process
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat((
                    cur_input_embeds[:img_start_token[batch_idx]],
                    cur_image_features,
                    cur_input_embeds[img_start_token[batch_idx] + cur_image_features.shape[0]:]
                ))
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_generation_idx += 1
            else:
                raise Exception("num_input_images != 0 and num_generated_images != 0 not implemented")

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels,image_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    #     if model_args.mm_use_im_patch_token:
    #         tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))

    #     if model_args.mm_use_im_start_end:
    #         num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))

    #         if num_new_tokens > 0:
    #             input_embeddings = self.get_input_embeddings().weight.data
    #             output_embeddings = self.get_output_embeddings().weight.data

    #             input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)
    #             output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)

    #             input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #             output_embeddings[-num_new_tokens:] = output_embeddings_avg

    #         if model_args.tune_mm_mlp_adapter:
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = True
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False

    #         if model_args.pretrain_mm_mlp_adapter:
    #             mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
    #             embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
    #             assert num_new_tokens == 2
    #             if input_embeddings.shape == embed_tokens_weight.shape:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
    #             elif embed_tokens_weight.shape[0] == num_new_tokens:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight
    #             else:
    #                 raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
    #     elif model_args.mm_use_im_patch_token:
    #         if model_args.tune_mm_mlp_adapter:
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = False
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False
