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


NoneType=type(None)
from typing import List, Optional, Tuple, Union, Dict

from httpx import get
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss, MSELoss


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM_ImgGen(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Dict[str, torch.Tensor]]=None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        img_token_start: Optional[List[torch.LongTensor]] = [None],
        **loss_kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
      
        img_row_indices =None
        img_col_indices = None

        #print(f"inputs_embeds: {inputs_embeds}")
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_labels
            ) = self.my_prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                img_token_start,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()
        loss = None
        num_patches=self.get_model().vision_tower_gen.num_patches

        image_loss=None
        if self.config.mm_projector_head_output_size!=self.config.mm_gen_hidden_size : #VQ output
            image_loss=nn.CrossEntropyLoss(reduction='none')
        elif self.config.image_loss=='cosine':
            image_loss=nn.CosineSimilarity(dim=-1)
        elif self.config.image_loss=='mse':
            image_loss=nn.MSELoss(reduction='none')
        

        if labels is not None:
            # 将 img_token_start 转为张量，如果是 None 则用 -1 占位
            start_positions = torch.tensor([pos if pos is not None else -1 for pos in img_token_start], dtype=torch.long, device=inputs_embeds.device)
            
            # 创建一个掩码，标记有效的起始位置
            valid_mask = start_positions >= 0

            # 筛选出有效的 batch 索引和对应的开始位置
            batch_indices = valid_mask.nonzero(as_tuple=True)[0]
            start_indices = start_positions[valid_mask]

            # 使用高级索引从 inputs_embeds 中抽取 img_token_start 到 img_token_start + seq_len 的部分
            img_row_indices = batch_indices.unsqueeze(1)
            img_col_indices = start_indices.unsqueeze(1) + torch.arange(num_patches,device=start_indices.device).unsqueeze(0)

            #get img embedding
            img_col_indices-=1
            img_embed_outputs = hidden_states[img_row_indices, img_col_indices]
    
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            projected_embed = self.get_model().mm_projector_head(img_embed_outputs)

            if isinstance(image_loss, nn.CrossEntropyLoss):
       
                patch_len=image_labels.shape[1]
                batch_size=image_labels.shape[0]
                image_labels = image_labels.view(-1)
                projected_embed=projected_embed.view(-1, self.config.mm_projector_head_output_size)
                # Enable model parallelism
                image_labels = image_labels.to(projected_embed.device)
                loss_gen = image_loss(projected_embed, image_labels)
                loss_gen=loss_gen.view(batch_size, patch_len)
            else:
                loss_gen = image_loss(projected_embed, image_labels)
                if self.config.image_loss=='cosine':
                    loss_gen = 1 - loss_gen
            padding_mask=torch.ones((loss_gen.shape),device=loss_gen.device)
            padding_mask[-1]=0
            loss_gen*= padding_mask
            loss_gen=loss_gen.mean()
            print(f"loss_gen: {loss_gen}")
            alpha=getattr(self.config, 'alpha', 0.1)
            loss += alpha*loss_gen
 
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[Dict[str, torch.Tensor]]=None,
        image_sizes: Optional[torch.Tensor] = None,
        img_token_start: Optional[List[torch.LongTensor]] = [None],
        input_img_features=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # if inputs_embeds is None:
        #     (
        #         input_ids,
        #         position_ids,
        #         attention_mask,
        #         past_key_values,
        #         inputs_embeds,
        #         labels,
        #         image_labels
        #     ) = self.my_prepare_inputs_labels_for_multimodal(
        #         input_ids,
        #         position_ids,
        #         attention_mask,
        #         past_key_values,
        #         labels,
        #         images,
        #         img_token_start,
        #         image_sizes
        #     )
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_labels
            ) = self.my_prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                img_token_start,
                image_sizes=image_sizes,
                input_img_features=input_img_features
            )
            # (
            #     inputs,
            #     position_ids,
            #     attention_mask,
            #     _,
            #     inputs_embeds,
            #     _
            # ) = self.my_prepare_inputs_labels_for_multimodal(
            #     inputs,
            #     position_ids,
            #     attention_mask,
            #     None,
            #     None,
            #     images,
            #     image_sizes=image_sizes
            # )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        kwargs['output_hidden_states'] = True
        #print(inputs_embeds)
        outputs=super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict_in_generate=True,
            **kwargs
        )
        #print(f"outputs_keys: {outputs.keys()}")
        #print(f"outputs: {outputs}")
        generated_tokens=outputs['sequences']
        hidden_states=outputs['hidden_states']
        #print(f"generated_tokens: {generated_tokens}")
        # 返回结果
        return {
            "generated_tokens": generated_tokens,
            "hidden_states": hidden_states,  # None if `<image>` not found
        }


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     images: Optional[Dict[str, torch.Tensor]]=None,
    #     image_sizes: Optional[torch.Tensor] = None,
    #     img_token_start: Optional[List[torch.LongTensor]] = [None],
    #     input_img_features=None,
    #     **kwargs,
    # )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # print(position_ids)
        # print(attention_mask) both is none in inference


        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_ImgGen)
