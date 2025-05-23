import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import SiglipImageProcessor, SiglipVisionModel,SiglipVisionConfig
from torchvision import transforms
from PIL import Image
from .vq_model import VQ_models

class VQArgs():
    def __init__(self):
        self.image_path = "/public_data/jihai/data/multimodalout/smart_watch_image_test/1.png"
        self.vq_model = "VQ-16"
        self.vq_ckpt = '/public_data/jihai/tmp/vq_ds16_c2i.pt'
        self.codebook_size = 16384
        self.codebook_embed_dim = 8
        self.image_size = 256
        self.seed = 0

class VQImageProcessor:
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # 调整尺寸
            transforms.ToTensor()                        # 转换为Tensor并自动归一化到[0,1
        ])
        self.image_mean=[0.5, 0.5, 0.5]

    def preprocess(self, image,return_tensors='pt'):
        image = self.transform(image)
        image=2.0*image-1.0
        # 添加批次维度并返回
        image=image.unsqueeze(0)
        out={'pixel_values':image}
        return out


class VQVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.vq_cfg= VQArgs()
        self.args=args
        #self.select_layer = args.mm_vision_select_layer
        #self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
           self.cfg_only = self.vq_cfg

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = VQImageProcessor(self.vq_cfg.image_size)
        self.vision_tower = VQ_models[self.vq_cfg.vq_model](codebook_size=self.vq_cfg.codebook_size,
                                                                       codebook_embed_dim=self.vq_cfg.codebook_embed_dim)
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
   
        checkpoint = torch.load(self.vq_cfg.vq_ckpt, map_location="cpu")
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        self.vision_tower.load_state_dict(model_weight)
        del checkpoint

        if self.args.vision_tower_permutation_path is None:
            self.permutation=nn.Identity()
        else:
            print(f"load vision permutation at {self.args.vision_tower_permutation_path}")
            state=torch.load(self.args.vision_tower_permutation_path)
            self.permutation=Affine(self.hidden_size)
            self.permutation.load_state_dict(state)
            self.permutation.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            image_indices=[]
            for image in images:
                latent, _, [_, _, indices] = self.vision_tower.encode(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                latent = latent.to(image.dtype)
                latent=latent.permute(0,2,3,1) #b,h,w,c
                latent=latent.view(latent.shape[0],-1,latent.shape[-1]) #b,seq_len,c
                latent = self.permutation(latent)
                image_features.append(latent)
                image_indices.append(indices)
        else:
          
            image_features, _, [_, _, image_indices] = self.vision_tower.encode(images.to(device=self.device, dtype=self.dtype))
            image_features = image_features.to(images.dtype)
            image_features=image_features.permute(0,2,3,1) #b,h,w,c
            image_features=image_features.view(image_features.shape[0],-1,image_features.shape[-1]) #b,seq_len,c
            image_features = self.permutation(image_features)

        return (image_features,image_indices)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 8

    @property
    def num_patches_per_side(self):
        return 16

    @property
    def num_patches(self):
        return 256


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args=args
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name,cache_dir=self.args.vision_tower_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map,cache_dir=self.args.vision_tower_path)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        
        self.register_buffer('A', torch.zeros(dim,dim))  # register_buffer 确保 A 不可训练
        self.register_buffer('b', torch.zeros(dim))  # 随机平移向量 b，不可训练

    def forward(self, x, *args, **kwargs):
        seq_len=x.shape[1]
        x=x.reshape(-1, self.A.shape[1])
        y=torch.matmul(x,self.A) + self.b
        y=y.view(-1, seq_len, self.A.shape[0])
        return y

class SigLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args=args

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name,cache_dir=self.args.vision_tower_path)
        self.vision_tower=SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map,cache_dir=self.args.vision_tower_path)
        self.vision_tower.requires_grad_(False)

        if self.args.vision_tower_permutation_path is None:
            self.permutation=nn.Identity()
        else:
            print(f"load vision permutation at {self.args.vision_tower_permutation_path}")
            state=torch.load(self.args.vision_tower_permutation_path)
            self.permutation=Affine(self.vision_tower.config.hidden_size)
            self.permutation.load_state_dict(state)
            self.permutation.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)

                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = self.permutation(image_feature)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = self.permutation(image_features)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2




class SyntheticVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.args=args
        self.hidden_dim=7

        

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            #raise Exception("Model loading is delayed and unfreeze_mm_vision_tower is not set.")
            pass
            

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = nn.Identity()
        state=torch.load(self.args.vision_tower_path)

        self.vision_tower=Affine(7)
        self.vision_tower.load_state_dict(state)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward_inner(self, x: torch.Tensor):
        #padded_x = F.pad(x, (0, self.hidden_dim - self.mm_dim))
        #return torch.matmul(padded_x,self.A)
        return self.vision_tower(x)

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature=self.forward_inner(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature.to(image.dtype))
                return image_features
        else:
            image_features=self.forward_inner(images.to(device=self.device, dtype=self.dtype))
            return image_features.to(images.dtype)

  

    @property
    def dtype(self):
        return self.vision_tower.A.dtype

    @property
    def device(self):
        return self.vision_tower.A.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.hidden_dim
    
    @property
    def num_patches(self):
        return 6
       




class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
