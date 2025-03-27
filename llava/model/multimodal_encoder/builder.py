import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2,SyntheticVisionTower,SigLIPVisionTower
# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence, List

# @dataclass
# class VisionArguments:
#     vision_tower: Optional[str] = field(default=None)
#     vision_tower_path: Optional[str] = field(default=None)
#     mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
#     mm_use_im_start_end: bool = field(default=False)
#     mm_use_im_patch_token: bool = field(default=True)
#     mm_patch_merge_type: Optional[str] = field(default='flat')
#     mm_vision_select_feature: Optional[str] = field(default="patch")

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "synthetic" in vision_tower.lower():
        return SyntheticVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower.lower():
        return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

# def build_vision_tower_gen(vision_tower_cfg, **kwargs):
#     vision_tower_inverse_cfg=VisionArguments(
#         vision_tower=vision_tower_cfg.vision_tower_inverse,
#         vision_tower_path=vision_tower_cfg.vision_tower_inverse_path,
#         mm_vision_select_layer=vision_tower_cfg.mm_vision_select_layer,
#         mm_use_im_start_end=vision_tower_cfg.mm_use_im_start_end,
#         mm_use_im_patch_token=vision_tower_cfg.mm_use_im_patch_token,
#         mm_patch_merge_type=vision_tower_cfg.mm_patch_merge_type,
#         mm_vision_select_feature=vision_tower_cfg.mm_vision_select_feature
#     )
#     vision_tower = getattr(vision_tower_inverse_cfg, 'mm_vision_tower', getattr(vision_tower_inverse_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_inverse_cfg, 's2', False)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_inverse_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_inverse_cfg, **kwargs)
#     if "synthetic" in vision_tower.lower():
#         return SyntheticVisionTower(vision_tower, args=vision_tower_inverse_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')
         