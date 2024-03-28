# CLIP encodes the image once, hence each layer has one attention map
# LLaMA encodes the sentence multiple times, and with only queries for generation,
# hence the latter attention maps of the same layer
# have the shape [B, H, 1, KV_LEN]
# Hence we create once attention matrix for one layer by concating everything
# `rollout(...)` copied over from 
# https://github.com/sayakpaul/vit-explain/blob/4f92628ed4b5109f43febd2976f688e585baa44b/vit_rollout.py
# Thanks @sayakpaul!

import torch
import numpy as np
from torchvision.utils import save_image
import cv2
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

def merge_images(image_batch, size):
    h,w = image_batch.shape[1], image_batch.shape[2]
    c = image_batch.shape[3]
    img = np.ones((int(h*size[0]), int(w*size[1]), c))
    for idx, im in enumerate(image_batch):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im
    return img

def calculate_modality_indices(inputs_emb_modalities, bsz=1, seq_len=None):
    # bsz, num_heads, q_len, kv_len = attn_weights.size()
    
    if seq_len == None:
        raise ValueError("`seq_len` should be a positive integer, found None.")
    
    image_attn_mask = torch.zeros(bsz, seq_len)
    video_attn_mask = torch.zeros(bsz, seq_len)
    gen_text_attn_mask = torch.zeros(bsz, seq_len)
    
    mask_map = dict(
        text=gen_text_attn_mask,
        image=image_attn_mask,
        video=video_attn_mask
    )
    
    modalities_buffer = inputs_emb_modalities
    # List[List[Dict['modality': num_tokens]]]
    
    for example_idx in range(len(modalities_buffer)):
        example_buffer = modalities_buffer[example_idx]
        running_tok_idx = 0
        for chunk_idx in range(len(example_buffer)):
            chunk_modality = list(example_buffer[chunk_idx].keys())[0]
            chunk_tokens = list(example_buffer[chunk_idx].values())[0]
            mask_map[chunk_modality][example_idx, running_tok_idx : running_tok_idx + chunk_tokens] = 1
            running_tok_idx += chunk_tokens
    
    
    return image_attn_mask, video_attn_mask, gen_text_attn_mask

def combine_attention(layer_attn_list):
    final_attn = list()
    max_len = layer_attn_list[-1].shape[-1]
    
    num_generated_tokens = 0
    
    for attn in layer_attn_list:
        if attn.shape[-2] == 1:
            num_generated_tokens += 1
        curr_kv_len = attn.shape[-1]
        final_attn.append(F.pad(attn, (0, max_len - curr_kv_len, 0, 0, 0, 0, 0, 0)))
    
    # print(num_generated_tokens)
    return torch.cat(final_attn, dim=-2), num_generated_tokens



def combine_all_layers(attns):
    for key in attns.keys():
        if key.startswith("llama"):
            attns[key] = combine_attention(attns[key])
    return attns


def compute_token_sims(MODEL, EXP1, EXP2, q1=None, q2=None, ans=None, has_modality=True, modality="image"):
    payload1 = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP1}.pt")
    modalities1 = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP1}_modalities.pt")
    
    payload2 = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP2}.pt")
    modalities2 = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP2}_modalities.pt")
    
    image_sims = list()
    text_sims = list()
    
    for layer_idx in range(32):
        _layer_tokens1 = payload1[f"llama_acts_{layer_idx}"]
        layer_tokens1 = torch.cat(_layer_tokens1, dim=-2)
        # for item in layer_tokens1:
        #     print(item.shape)
        
        # print(f"first seq len: {layer_tokens1.shape}")
        
        img_mask1, vid_mask1, text_mask1 = calculate_modality_indices(modalities1, seq_len=layer_tokens1.shape[-2])
        
        _layer_tokens2 = payload2[f"llama_acts_{layer_idx}"]
        layer_tokens2 = torch.cat(_layer_tokens2, dim=-2)
        
        # print(f"second seq len: {layer_tokens2.shape}")
        
        
        img_mask2, vid_mask2, text_mask2 = calculate_modality_indices(modalities2, seq_len=layer_tokens2.shape[-2])
        
        assert img_mask1.sum() == img_mask2.sum(), "both models have different "\
            "number of image tokens, check model args and try again"
        

        # print(layer_tokens1.shape)
        # print(img_mask1.shape)
        
        # THIS WORKS ONLY BECAUSE BATCH SIZE = 1! Be careful with indexing tensors
        # using tensors
        image_tokens1 = layer_tokens1[img_mask1.bool()] 
        image_tokens2 = layer_tokens2[img_mask2.bool()] 
        # output shape is [N, D] and not [B, N, C] since tensor indexing 
        # flattens the indexed dimensions
        
        # image_tokens1 = layer_tokens1[img_mask1.unsqueeze(-1).expand(1, 1, layer_tokens1.shape[-1]).bool()]
        # image_tokens2 = layer_tokens2[img_mask2.unsqueeze(-1).expand(1, 1, layer_tokens1.shape[-1]).bool()]
        # print()
        # print(f"Magnitude difference in image tokens: {(image_tokens2 - image_tokens1).abs().sum():.4f}")
        # print(f"Avg per value mag diff in image tokens: {(image_tokens2 - image_tokens1).abs().sum() / image_tokens1.numel():.4f}")
        
        prompt_tokens1 = layer_tokens1[0, :31]
        prompt_tokens2 = layer_tokens2[0, :31]
        
        # print(f"Magnitude difference in system prompt tokens: {(prompt_tokens2 - prompt_tokens1).abs().sum():.4f}")
        # print(f"Avg per value mag diff in image tokens: {(prompt_tokens2 - prompt_tokens1).abs().sum() / prompt_tokens1.numel():.4f}")
        
        
        text_tokens1 = torch.cat(_layer_tokens1[1:], dim=-2) # selects only the generated tokens
        text_tokens2 = torch.cat(_layer_tokens2[1:], dim=-2)

        
        min_len = min(text_tokens1.shape[-2], text_tokens2.shape[-2])
        
        text_tokens1 = text_tokens1[..., :min_len, :]
        text_tokens2 = text_tokens2[..., :min_len, :]
        
        image_sims.append(torch.mean(torch.nn.functional.cosine_similarity(image_tokens1, image_tokens2, dim=-1)).cpu().numpy().item())
        text_sims.append(torch.mean(torch.nn.functional.cosine_similarity(text_tokens1, text_tokens2, dim=-1)).cpu().numpy().item())
    
    return image_sims, text_sims
        
        
        

def plot_attn_vis(MODEL, EXP, q=None, ans=None, has_modality=True, modality="image"):
    attns = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP}.pt")
    modalities = torch.load(f"/data/data0/akane/attn_tensors/{MODEL}_{EXP}_modalities.pt")
    
    all_images = list()
    img_attns = list()
    gen_text_attns = list()
    prompt_attns = list()
    question_attns = list()
    
    for layer_idx in range(32):
        combined_attn, num_generated_tokens = combine_attention(attns[f"llama_attn_{layer_idx}"])
        pooled_combined_attn = torch.mean(combined_attn, dim=1)
        # print(pooled_combined_attn.shape)
        first_token_attn_wrt_gen_tokens = pooled_combined_attn[..., -num_generated_tokens:, :1].cpu()
        gen_text_attn_wrt_gen_tokens = pooled_combined_attn[..., -num_generated_tokens:, -num_generated_tokens:].cpu()
        
        bos_len = 1
        img_len = 576
        sys_prompt_len = 31
        USER_str_len = 5
        question_len = 7
        
        # CASE 1: BOS + <SysPrompt> + "USER: " + <image tokens> + <Question>
        # prompt_attn_wrt_gen_tokens = \
        #     pooled_combined_attn[..., -num_generated_tokens:, bos_len:bos_len + sys_prompt_len].cpu()
        # question_attn_wrt_gen_tokens =\
        #     torch.cat(
        #         [pooled_combined_attn[..., -num_generated_tokens:, bos_len + sys_prompt_len : bos_len + sys_prompt_len + USER_str_len].cpu(),
        #          pooled_combined_attn[..., -num_generated_tokens:, bos_len + sys_prompt_len + USER_str_len + img_len : bos_len + sys_prompt_len + USER_str_len + img_len + question_len].cpu()],
        #     dim=-1)
        
        # Case 2: BOS + "USER: " + <image tokens> + <Question>
        # question_attn_wrt_gen_tokens =\
        #     torch.cat(
        #         [pooled_combined_attn[..., -num_generated_tokens:, bos_len : bos_len + USER_str_len].cpu(),
        #          pooled_combined_attn[..., -num_generated_tokens:, bos_len + USER_str_len + img_len : bos_len + USER_str_len + img_len + question_len].cpu()],
        #     dim=-1)
        
        # # Case 3: BOS + "USER: " + <image tokens> + <Question> + <SysPrompt>
        prompt_attn_wrt_gen_tokens =\
            pooled_combined_attn[..., -num_generated_tokens:, bos_len + USER_str_len + img_len + question_len : bos_len + USER_str_len + img_len + question_len + sys_prompt_len ].cpu()
        question_attn_wrt_gen_tokens =\
            torch.cat(
                [pooled_combined_attn[..., -num_generated_tokens:, bos_len : bos_len + USER_str_len].cpu(),
                 pooled_combined_attn[..., -num_generated_tokens:, bos_len + USER_str_len + img_len : bos_len + USER_str_len + img_len + question_len].cpu()],
            dim=-1)
        
        
        
        if has_modality:
            img_mask, vid_mask, text_mask = calculate_modality_indices(modalities, seq_len=combined_attn.shape[-1])
            
            # print(img_mask[0])

            if modality == "image":
                img_attn_wrt_gen_tokens = pooled_combined_attn[..., -num_generated_tokens:, img_mask[0].to(torch.bool)].cpu()
                modality_tokens = int(torch.sum(img_mask).cpu().item())
            else:
                img_attn_wrt_gen_tokens = pooled_combined_attn[..., -num_generated_tokens:, vid_mask[0].to(torch.bool)].cpu()
                modality_tokens = int(torch.sum(vid_mask).cpu().item())

            num_tokens = 4
            # print()
            # print(img_attn_wrt_gen_tokens[0, :5, :10])
            # print()
            # print(pooled_combined_attn[0,  -num_generated_tokens: -num_generated_tokens + 5, :10])
            
            img_attn = img_attn_wrt_gen_tokens.float()
            _avg_img_attn_wrt_gen = img_attn.sum().item()/ num_generated_tokens
            img_attns.append(_avg_img_attn_wrt_gen)
            
            gen_text_attn = gen_text_attn_wrt_gen_tokens.float() # to account for causal mask
            _avg_gen_attn_wrt_gen = gen_text_attn.sum().item() / num_generated_tokens
            gen_text_attns.append(_avg_gen_attn_wrt_gen)
            
            # _avg_prompt_attn_wrt_gen = 0.
            prompt_attn = prompt_attn_wrt_gen_tokens.float()
            _avg_prompt_attn_wrt_gen = prompt_attn_wrt_gen_tokens.sum().item() / num_generated_tokens
            prompt_attns.append(_avg_prompt_attn_wrt_gen)
            
            _avg_first_tokens_attn_wrt_gen = first_token_attn_wrt_gen_tokens.sum().item() / num_generated_tokens

            question_attn = question_attn_wrt_gen_tokens.float()
            _avg_question_attn_wrt_gen = question_attn.sum().item() / num_generated_tokens
            question_attns.append(_avg_question_attn_wrt_gen)
            
            
            print(f"****** Layer {layer_idx}")
            print(f"Total attn of first token wrt gen tokens (avged per gen query) for layer {layer_idx}: {_avg_first_tokens_attn_wrt_gen}")
            print(f"Total image attn by gen tokens (avged per gen query) for layer {layer_idx}: {_avg_img_attn_wrt_gen}")
            print(f"Total question text attn by gen tokens (avged per gen query) for layer {layer_idx}: {_avg_question_attn_wrt_gen}")
            print(f"Total prompt text attn by gen tokens (avged per gen query) for layer {layer_idx}: {_avg_prompt_attn_wrt_gen}")
            print(f"Total gen text attn by gen tokens (avged per gen query) for layer {layer_idx}: {_avg_gen_attn_wrt_gen}")
            print()
            
            
            step = modality_tokens // num_tokens
            avg_img_attn = torch.nn.functional.avg_pool1d(img_attn_wrt_gen_tokens.float(), kernel_size=step, stride=step) * step
            # avg_img_attn = img_attn_wrt_gen_tokens.float()
        else:
            avg_img_attn = torch.zeros(list(gen_text_attn_wrt_gen_tokens.shape)[:-1] + [0,])
          
        pooled_combined_attn = torch.concat([avg_img_attn, gen_text_attn_wrt_gen_tokens], dim=-1)
        pooled_combined_attn = pooled_combined_attn.permute(1,2,0).cpu().numpy()
        
        # print(pooled_combined_attn[:20, :20])
        
        score_map = torch.tensor(np.uint8(255 * pooled_combined_attn)).to(torch.float32).permute(2, 0, 1) / 255.
        # print(score_map[..., -10:])
        # plt.plot(score_map.numpy()[0])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(score_map.permute(1, 2 ,0).numpy(), cmap='viridis', interpolation='nearest')
        os.makedirs(f"/data/data0/akane/attn_vis/{MODEL}_{EXP}", exist_ok=True)
        plt.savefig(f"/data/data0/akane/attn_vis/{MODEL}_{EXP}/attn_wrt_gen_tokens_{layer_idx}.png",  bbox_inches='tight', pad_inches=0.0)
        
        all_images.append(score_map.permute(1, 2 ,0).numpy())
        
    print(f"####### Avg image attn for all layers: {sum(img_attns) / 32}")
    print(f"####### Avg gen text attn for all layers: {sum(gen_text_attns) / 32}")
    print(f"####### Avg prompt attn for all layers: {sum(prompt_attns) / 32}")
    print(f"####### Avg question attn for all layers: {sum(question_attns) / 32}")
        
    with open(f"/data/data0/akane/attn_vis/{MODEL}_{EXP}/_output.txt", "w+") as f:
        if q is not None:
            f.write(q)
                        
        if ans is not None:
            _output = [ans[i : i + 80] + "\n" for i in range(0, len(ans), 80)]
            f.write("\n>>> ")
            f.writelines(_output)
    grid = merge_images(np.stack(all_images, axis=0), [4, 8])
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.savefig(f"/data/data0/akane/attn_vis/{MODEL}_{EXP}/_grid.png", bbox_inches='tight', pad_inches=0.)




def rollout(attentions, discard_ratio=0.8, head_fusion="min"):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    