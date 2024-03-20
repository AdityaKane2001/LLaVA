import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model.llava_arch import ModalityBuffer

from PIL import Image
import os

import requests
from PIL import Image
from io import BytesIO
import re

# torch.cuda.set_device("cuda:0")

import transformers

transformers.set_seed(42)
torch.cuda.manual_seed_all(42)

from get_attn_maps import plot_attn_vis



def add_forward_hooks(model, cache, key_prefix=""):
    def get_llama_attn_maps(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):
            cache[key_prefix + name].append(output[1].detach())
        return hook
    
    def get_llama_acts(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):
            cache[key_prefix + name].append(output[0].detach())
        return hook

    def get_clip_attn_maps(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):            
            cache[key_prefix + name].append(output[1].detach())
        return hook
    
    all_hooks = list()
    
    # add hooks from llama LLM
    for block_idx in range(len(model.model.layers)):
        all_hooks.append(model.model.layers[block_idx].self_attn.register_forward_hook(
                get_llama_attn_maps(f"llama_attn_{block_idx}")
            )
        ) 
        all_hooks.append(
            model.model.layers[block_idx].register_forward_hook(
                get_llama_acts(f"llama_acts_{block_idx}")
            )
        ) 
    
    # add hooks from image tower
    # for block_idx in range(len(model.model.image_tower.image_tower.encoder.layers)):
    #     all_hooks.append(model.model.image_tower.image_tower.encoder.layers[block_idx].self_attn.register_forward_hook(
    #             get_clip_attn_maps(f"clip_attn_{block_idx}")
    #         )
    #     )
    
    return model, all_hooks


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    
    
    checkpoint2name = {
        "/data/data0/akane/dupl-glbltok-pretrained-grllava-v1.5-7b/checkpoints": "dupl-glbltok-pretrained-grllava",
        "/data/data0/akane/dupl-glbltok-scratch-grllava-v1.5-7b/checkpoints": "dupl-glbltok-scratch-grllava",
        "/data/data0/akane/noscaling-residual-grllava-pretrained-v1.5-7b/checkpoints": "noscaling-residual-grllava-pretrained",
        "/data/data0/akane/residual-grllava-pretrained-v1.5-7b/checkpoints": "residual-grllava-pretrained",
        "/data/data1/akane/grllava-pretrained-v1.5-7b/checkpoints": "grllava-pretrained",
        "liuhaotian/llava-v1.5-7b": "llava-v1.5-7b"
    }
    
    MODEL = checkpoint2name[args.model_path] # one of "image", "video"
    EXP = "man_ironing_noprompt" # experiment description
    # Model
    disable_torch_init()

    model_name = "llava-v1.5-7b" # get_model_name_from_path(args.model_path)
    # print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,
    )
    
    hook_cache = dict()
    model, all_hooks = add_forward_hooks(model, hook_cache)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    with torch.inference_mode():
        
        # print(model.get_vision_tower().vision_tower.vision_model.encoder.layers[0].mlp.fc1.weight)
        
        # model.get_model().granular_mm_projector[0].weight.data = model.get_model().mm_projector[0].weight.data
        # model.get_model().granular_mm_projector[0].bias.data = model.get_model().mm_projector[0].bias.data
        
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    os.makedirs(f"/data/data0/akane/attn_tensors/", exist_ok=True)
    torch.save(hook_cache, f"/data/data0/akane/attn_tensors/{MODEL}_{EXP}.pt")
    torch.save(ModalityBuffer.inputs_emb_modalities, f"/data/data0/akane/attn_tensors/{MODEL}_{EXP}_modalities.pt")
    for hook in all_hooks:
        hook.remove()
    
    print(output_ids)
    try:
        output_ids.keys()
    except:
        print(output_ids.shape)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    
    plot_attn_vis(MODEL, EXP, q=qs, ans=outputs, has_modality=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # /data/data0/akane/dupl-glbltok-pretrained-grllava-v1.5-7b/checkpoints
    # /data/data0/akane/dupl-glbltok-scratch-grllava-v1.5-7b/checkpoints
    # /data/data0/akane/noscaling-residual-grllava-pretrained-v1.5-7b/checkpoints
    # /data/data0/akane/residual-grllava-pretrained-v1.5-7b/checkpoints
    # /data/data1/akane/grllava-pretrained-v1.5-7b/checkpoints
    # liuhaotian/llava-v1.5-7b
    
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/home/akane38/LLaVA/llava/serve/examples/extreme_ironing.jpg")
    parser.add_argument("--query", type=str, default="") # What is odd about this image?
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
