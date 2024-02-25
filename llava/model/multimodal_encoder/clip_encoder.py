import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from .merge import bipartite_soft_matching


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        try: 
            self.use_additional_adapter = args.mm_vision_use_additional_adapter
            self.use_global_tokens = args.mm_vision_use_global_tokens
            self.use_granular_tokens = args.mm_vision_use_granular_tokens
            self.granular_layers = args.mm_vision_granular_select_layers
            self.granular_tokens_per_layer = args.mm_vision_granular_tokens_per_layer
            self.granular_tokens_strategy = args.mm_vision_granular_tokens_strategy
            print("Granular tokens config loaded!")
            print(f"{self.use_additional_adapter=}")
            print(f"{self.use_global_tokens=}")
            print(f"{self.use_granular_tokens=}")
            print(f"{self.granular_layers=}")
            print(f"{self.granular_tokens_per_layer=}")
            print(f"{self.granular_tokens_strategy=}")
        except:
            self.use_additional_adapter = False
            self.use_global_tokens = False
            self.use_granular_tokens = False
            self.granular_layers = ""
            self.granular_tokens_per_layer = 0
            self.granular_tokens_strategy = None
            print("Granular tokens config not found, falling back to not "
                  "using granular tokens.")
            

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

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
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

    def compress_granular_features(self, tokens, target_num, strategy="pool"):
        num_tokens = tokens.shape[-2]
        assert num_tokens % target_num, \
        f"Compressed tokens per layer should divide number "\
        f"of tokens, got {tokens.shape[-2]=}, {target_num=}."
        
        if strategy == "pool":
            step = num_tokens // target_num
            compressed = F.avg_pool1d(tokens.transpose(1, 2), kernel_size=step).transpose(1, 2)
        else:
            raise ValueError(f"Token compression strategy not implemented: {strategy}")
        return compressed

    def granular_feature_select(self, image_forward_outs):
        features = list()
        layer_indices = list(map(lambda x: int(x), self.granular_layers.split()))
        for layer_idx in layer_indices:
            _feats = image_forward_outs.hidden_states[layer_idx]
            compressed_feats = self.compress_granular_features(
                _feats,
                target_num=self.granular_tokens_per_layer,
                strategy=self.granular_tokens_strategy
            )
            features.append(compressed_feats)
        features = torch.concat(features, dim=-2)
        return features
        
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                
                if self.use_additional_adapter and self.use_granular_tokens:
                    # print("Using granular tokens")                    
                    # append granular tokens to image_features
                    granular_feature = self.granular_feature_select(image_forward_out)
                    image_feature = torch.concat([granular_feature, image_feature], dim=-2)
                
                elif self.use_additional_adapter and self.use_global_tokens:
                    global_feature = self.feature_select(image_forward_out)
                    image_feature = torch.concat([global_feature, image_feature], dim=-2)

                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            
            if self.use_additional_adapter and self.use_granular_tokens:
                granular_features = self.granular_feature_select(image_forward_outs).to(images.dtype)
                image_features = torch.concat([granular_features, image_features], dim=-2)
            
            elif self.use_additional_adapter and self.use_global_tokens:
                global_features = self.feature_select(image_forward_outs).to(images.dtype)
                image_features = torch.concat([global_features, image_features], dim=-2)

            
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



class OldCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

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

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
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
