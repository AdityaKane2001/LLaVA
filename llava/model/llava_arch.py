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

import torch
import torch.nn as nn
from torch.nn import functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class ModalityBuffer:
    inputs_emb_modalities = None
    num_text_tokens = 0
    num_image_tokens = 0
    num_video_tokens = 0
    init_seq_len = 0
    
    @staticmethod
    def reset():
        ModalityBuffer.inputs_emb_modalities = None
        ModalityBuffer.num_text_tokens = 0
        ModalityBuffer.num_image_tokens = 0
        ModalityBuffer.num_video_tokens = 0
        ModalityBuffer.init_seq_len = 0
    
    @staticmethod
    def calculate_modality_indices(bsz, seq_len):
        
        # bsz, num_heads, q_len, kv_len = attn_weights.size()
        
        image_attn_mask = torch.zeros(bsz, seq_len)
        video_attn_mask = torch.zeros(bsz, seq_len)
        text_attn_mask = torch.zeros(bsz, seq_len)
        
        mask_map = dict(
            text=text_attn_mask,
            image=image_attn_mask,
            video=video_attn_mask
        )
        
        modalities_buffer = ModalityBuffer.inputs_emb_modalities
        # List[List[Dict['modality': num_tokens]]]
        
        for example_idx in range(len(modalities_buffer)):
            example_buffer = modalities_buffer[example_idx]
            running_tok_idx = 0
            for chunk_idx in range(len(example_buffer)):
                chunk_modality = list(example_buffer[chunk_idx].keys())[0]
                chunk_tokens = list(example_buffer[chunk_idx].values())[0]
                mask_map[chunk_modality][example_idx, running_tok_idx : running_tok_idx + chunk_tokens] = 1
                running_tok_idx += chunk_tokens
        
        _num_img_tok = image_attn_mask.sum().item()
        _num_vid_tok = video_attn_mask.sum().item()
        _num_text_tok = text_attn_mask.sum().item()
        
        if ModalityBuffer.init_seq_len < seq_len:
            ModalityBuffer.init_seq_len = seq_len
        
        if ModalityBuffer.num_text_tokens < _num_text_tok:
            ModalityBuffer.num_text_tokens = _num_text_tok
        
        if ModalityBuffer.num_image_tokens < _num_img_tok:
            ModalityBuffer.num_image_tokens = _num_img_tok
        
        if ModalityBuffer.num_video_tokens < _num_vid_tok:
            ModalityBuffer.num_video_tokens = _num_vid_tok
                
        return image_attn_mask, video_attn_mask, text_attn_mask



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.granular_mm_projector = build_vision_projector(config)
            print("Built vision tower and projector!")
            
            if hasattr(config, "mm_vision_use_scaled_residual_granular_tokens") and config.mm_vision_use_scaled_residual_granular_tokens:
                self.granular_tokens_scaler = nn.Parameter(torch.zeros(config.mm_vision_num_tokens_per_layer), requires_grad=True)  
            
            if hasattr(config, "mm_vision_use_static_scaled_residual_granular_tokens") and config.mm_vision_use_static_scaled_residual_granular_tokens:
                self.granular_tokens_scaler = nn.Parameter(torch.zeros((1,)), requires_grad=True)            

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        
        mm_vision_num_tokens_per_layer = model_args.mm_vision_num_tokens_per_layer
        mm_vision_use_additional_adapter = model_args.mm_vision_use_additional_adapter
        mm_vision_use_pretrained_additional_adapter = model_args.mm_vision_use_pretrained_additional_adapter
        mm_vision_use_global_tokens = model_args.mm_vision_use_global_tokens
        mm_vision_use_granular_tokens = model_args.mm_vision_use_granular_tokens
        mm_vision_use_scaled_residual_granular_tokens = model_args.mm_vision_use_scaled_residual_granular_tokens
        mm_vision_use_static_scaled_residual_granular_tokens = model_args.mm_vision_use_static_scaled_residual_granular_tokens
        mm_vision_use_residual_scaler = model_args.mm_vision_use_residual_scaler
        mm_vision_use_static_residual_scaler = model_args.mm_vision_use_static_residual_scaler
        mm_vision_granular_tokens_per_layer = model_args.mm_vision_granular_tokens_per_layer
        mm_vision_granular_select_layers = model_args.mm_vision_granular_select_layers
        mm_vision_granular_tokens_strategy = model_args.mm_vision_granular_tokens_strategy

        self.config.mm_vision_tower = vision_tower

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

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        
        self.config.mm_vision_num_tokens_per_layer = mm_vision_num_tokens_per_layer
        self.config.mm_vision_use_additional_adapter = mm_vision_use_additional_adapter
        self.config.mm_vision_use_pretrained_additional_adapter = mm_vision_use_pretrained_additional_adapter
        self.config.mm_vision_use_global_tokens = mm_vision_use_global_tokens
        self.config.mm_vision_use_granular_tokens = mm_vision_use_granular_tokens
        self.config.mm_vision_use_scaled_residual_granular_tokens = mm_vision_use_scaled_residual_granular_tokens
        self.config.mm_vision_use_static_scaled_residual_granular_tokens = mm_vision_use_static_scaled_residual_granular_tokens
        self.config.mm_vision_use_residual_scaler = mm_vision_use_residual_scaler
        self.config.mm_vision_use_static_residual_scaler = mm_vision_use_static_residual_scaler
        self.config.mm_vision_granular_tokens_per_layer = mm_vision_granular_tokens_per_layer
        self.config.mm_vision_granular_select_layers = mm_vision_granular_select_layers
        self.config.mm_vision_granular_tokens_strategy = mm_vision_granular_tokens_strategy

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            self.granular_mm_projector = build_vision_projector(self.config)
            
            if hasattr(self.config, "mm_vision_use_residual_scaler") and self.config.mm_vision_use_residual_scaler:
                print("`mm_vision_use_residual_scaler` is set, initializing scaler")
                self.granular_tokens_scaler = nn.Parameter(torch.zeros(self.config.mm_vision_num_tokens_per_layer), requires_grad=True)
            else:
                print("mm_vision_use_residual_scaler not found or is False!")
            
            if hasattr(self.config, "mm_vision_use_static_scaled_residual_granular_tokens") and self.config.mm_vision_use_static_scaled_residual_granular_tokens:
                self.granular_tokens_scaler = nn.Parameter(torch.zeros((1,)), requires_grad=True)
                print("Creating static scaler")
            else:
                print("Static scaler config not found, skipping.")
            
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print("`pretrain_mm_mlp_adapter` is not None, loading weights into adapter")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            
            if self.config.mm_vision_use_additional_adapter and self.config.mm_vision_use_pretrained_additional_adapter:
                print("Additional adapter is set to be pretrained, loading pretrained weights to granular adapter")
                
                granular_mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.granular_mm_projector.load_state_dict(get_w(granular_mm_projector_weights, 'mm_projector'))
            else:
                print("Additional adapter is set to be scratch, skipping weight loading")
                
            
            # if self.config.mm_vision_use_additional_adapter and self.config.mm_vision_use_granular_tokens:
            #     granular_mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            #     def get_w(weights, keyword):
            #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            #     self.granular_mm_projector.load_state_dict(get_w(granular_mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

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
    
    def _compress_granular_features(self, tokens, target_num, strategy="pool"):
        num_tokens = tokens.shape[-2]
        assert num_tokens % target_num == 0, \
        f"Compressed tokens per layer should divide number "\
        f"of tokens, got {tokens.shape[-2]=}, {target_num=}."
        
        if strategy == "pool":
            step = num_tokens // target_num
            compressed = F.avg_pool1d(tokens.transpose(1, 2), kernel_size=step).transpose(1, 2)
        elif strategy == "uncompressed":
            return tokens
        else:
            raise ValueError(f"Token compression strategy not implemented: {strategy}")
        return compressed
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        
        if hasattr(self.config, "mm_vision_use_granular_tokens") and hasattr(self.config, "mm_vision_use_additional_adapter") \
            and self.config.mm_vision_use_granular_tokens and self.config.mm_vision_use_additional_adapter:
            # Use additional adapter and granular tokens
            
            num_tokens = image_features.shape[-2]
            granular_image_features = self.get_model().granular_mm_projector(image_features[..., :num_tokens//2, :])
            image_features = self.get_model().mm_projector(image_features[..., num_tokens//2:, :])
            image_features = torch.concat([granular_image_features, image_features], dim=-2)
        elif hasattr(self.config, "mm_vision_use_global_tokens") and hasattr(self.config, "mm_vision_use_additional_adapter") \
            and self.config.mm_vision_use_global_tokens and self.config.mm_vision_use_additional_adapter:
            # Use assitional adapter and global (duplicated) tokens
                
            num_tokens = image_features.shape[-2]
            granular_image_features = self.get_model().granular_mm_projector(image_features[..., :num_tokens//2, :])
            image_features = self.get_model().mm_projector(image_features[..., num_tokens//2:, :])
            image_features = torch.concat([granular_image_features, image_features], dim=-2)
        elif hasattr(self.config, "mm_vision_use_scaled_residual_granular_tokens") and hasattr(self.config, "mm_vision_use_additional_adapter") \
            and self.config.mm_vision_use_scaled_residual_granular_tokens and self.config.mm_vision_use_additional_adapter:
            # 1. Get granular tokens from ViT layers
            # 2. Pass them through second adapter
            # 3. Pool them
            # 4. Add them with scaling parameter to the original tokens
            
            # clip returns [(intermediate tokens 1, intermediate tokens 2, ...) | final layer outputs tokens]
            num_tokens = image_features.shape[-2]
            num_granular_layers = len(list(map(lambda x: int(x), self.config.mm_vision_granular_select_layers.split())))
            per_layer_tokens = num_tokens // (num_granular_layers + 1)

            original_image_tokens = image_features[..., num_granular_layers * per_layer_tokens:, :]
            
            granular_tokens = image_features[..., :num_granular_layers * per_layer_tokens, :]
            
            granular_image_features = self.get_model().granular_mm_projector(granular_tokens)
            # print(f"{granular_image_features.shape=}")
            
            compressed_feats = list()            
            for layer_idx in range(num_granular_layers):
                compressed = self._compress_granular_features(
                        tokens=granular_image_features[..., layer_idx * per_layer_tokens: (layer_idx + 1) * per_layer_tokens, :],
                        target_num=self.config.mm_vision_granular_tokens_per_layer,
                        strategy=self.config.mm_vision_granular_tokens_strategy
                    )
                compressed_feats.append(compressed)
                # print(f"{layer_idx=}; {compressed.shape=}")
            granular_image_features = torch.concat(compressed_feats, dim=-2)
            # print(f"{granular_image_features.shape=}")
            
            if self.config.mm_vision_use_residual_scaler:
                granular_image_features = granular_image_features * self.get_model().granular_tokens_scaler.view(1, -1, 1)
            else:
                pass
            
            image_features = self.get_model().mm_projector(original_image_tokens)
            
            image_features = image_features + granular_image_features
            # print(f"{image_features.shape=}")
            # image_features = torch.concat([granular_image_features, image_features], dim=-2)
        elif hasattr(self.config, "mm_vision_use_static_scaled_residual_granular_tokens") and hasattr(self.config, "mm_vision_use_additional_adapter") \
            and self.config.mm_vision_use_static_scaled_residual_granular_tokens and not self.config.mm_vision_use_additional_adapter:
            # Don't use additional adapter, just have one scaling parameter which 
            # scales the tokens from the earlier layers and adds them to the 
            # global tokens
            
            
            num_tokens = image_features.shape[-2]
            num_granular_layers = len(list(map(lambda x: int(x), self.config.mm_vision_granular_select_layers.split())))
            per_layer_tokens = num_tokens // (num_granular_layers + 1)

            original_image_tokens = image_features[..., num_granular_layers * per_layer_tokens:, :]
            
            granular_tokens = image_features[..., :num_granular_layers * per_layer_tokens, :]
            
            granular_tokens = granular_tokens * self.get_model().granular_tokens_scaler
            
            B, N, C = granular_tokens.shape

            granular_tokens = granular_tokens.view(B, num_granular_layers, N // num_granular_layers, C)
            granular_tokens = torch.mean(granular_tokens, dim=1)
            
            feats = self.get_model().mm_projector(granular_tokens + original_image_tokens)
            
            print(feats.shape)
            
            return  feats
        else:
            image_features = self.get_model().mm_projector(image_features)
        
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, inputs_emb_modalities=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1: # generation model
            if inputs_emb_modalities is not None:
                for example_idx in range(input_ids.shape[0]):
                    inputs_emb_modalities[example_idx].append({"text" : 1})
            else:
                inputs_emb_modalities = [[{"text": 1}] * input_ids.shape[0]]
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, inputs_emb_modalities


        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

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
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = [] # Initialize input embeddings tensor stack
        new_labels = []
        cur_image_idx = 0 # initialize current image index to 0
        new_input_embeds_modalities = list()# initialize modality buffer
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_new_input_embeds_modalities = list()
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # input does not have any images
                # so get features at the given
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_new_input_embeds_modalities.append({"text": cur_input_embeds.shape[0]})
                cur_image_idx += 1
                continue
            
            # The logic is as follows
            # 1. gather all text tokens and get their embeddings
            # 2. split them according to where image tokens are in between them
            # 3. stitch the final sequence by including image feats wherever image
            #    token is encountered
            # Thus, we append to modality buffer accordingly
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_input_embeds_modalities.append({"text": cur_input_embeds_no_im[i].shape[0]})
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds_modalities.append({"image": cur_image_features.shape[0]})
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_input_embeds_modalities.append(cur_new_input_embeds_modalities)
            new_labels.append(cur_new_labels)

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

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_input_embeds_modalities
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
