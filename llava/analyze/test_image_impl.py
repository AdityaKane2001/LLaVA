import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import transformers
from transformers.models.llama.modeling_llama import ModalityBuffer
from get_attn_maps import plot_attn_vis
from PIL import Image
import warnings

torch.cuda.set_device("cuda:7")

# FIXME: many warnings of loading non-meta weights to meta weights #############
# warnings.filterwarnings("ignore")

transformers.set_seed(42)
transformers.utils.logging.set_verbosity_error()
torch.set_warn_always(False)

QUESTION_BANK = {
    "suits": {
        "correct": "Is there a woman in this image? Answer in yes or no.",
        "wrong": "What is the capital of France? What is it famous for?"
    },
    "corgi": {
        "correct": "What do you see in the image?",
        "wrong": "What is the capital of France? What is it famous for?"
    },
    "stop": {
        "correct": "What should a driver do when they see this sign?",
        "wrong": "What is the capital of France? What is it famous for?"
    },
    "gencorgi": {
        "correct": "What is the animal in this image?",
        "wrong": "What is the capital of France? What is it famous for?"
    },
}


IMAGE_NAME2PATH = {
    "suits": "/home/akane38/LLaVA/llava/serve/examples/suits.jpeg",
    "corgi": "/home/akane38/LLaVA/llava/serve/examples/corgi.jpeg",
    "stop": "/home/akane38/LLaVA/llava/serve/examples/stop.jpeg",
    "gencorgi": "/home/akane38/LLaVA/llava/serve/examples/gendog.webp",
}

MODALITY = "image" # one of "image", "video"
IMAGE_NAME = "suits"
QUESTION_TYPE = "wrong"
DROP_LAYER_TYPE = "drop_tokens" # one of "drop_only_attn", "drop_tokens", "merge_unnmerge" or any other description
DROP_AT = 20
EXP = f"{IMAGE_NAME}_{QUESTION_TYPE}_{DROP_LAYER_TYPE}_dropat{DROP_AT}" # experiment description
HAS_MOD = True

def add_forward_hooks(model, cache, key_prefix=""):
    def get_llama_attn_maps(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):
            cache[key_prefix + name].append(output[1].detach())
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
    
    # add hooks from image tower
    # print(model.model.vision_tower) 
    for block_idx in range(len(model.model.vision_tower.vision_tower.vision_model.encoder.layers)):
        all_hooks.append(model.model.vision_tower.vision_tower.vision_model.encoder.layers[block_idx].self_attn.register_forward_hook(
                get_clip_attn_maps(f"clip_attn_{block_idx}")
            )
        )
    
    return model, all_hooks

def main():
    disable_torch_init()
    image = IMAGE_NAME2PATH[IMAGE_NAME] # '/home/akane38/LLaVA/llava/serve/examples/3.jpg'
    image = Image.open(image)
    inp = QUESTION_BANK[IMAGE_NAME][QUESTION_TYPE] #'Is there a woman in this image? Answer in yes or no.'
    model_path = 'liuhaotian/llava-v1.6-vicuna-7b'
    cache_dir = '../model/cache_dir'
    device = 'cuda:0'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir, 
                                        drop_at=DROP_AT, drop_layer_type=DROP_LAYER_TYPE)
    image_processor = processor
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    hook_cache = dict()
    model, all_hooks = add_forward_hooks(model, hook_cache)

    
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    # print(image_tensor)
    if type(image_tensor) is list:
        tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        tensor = image_tensor.to(model.device, dtype=torch.float16)
    key = ["image"]

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_attentions=True
        )
    
    torch.save(hook_cache, f"./attn_tensors/{MODALITY}_{DROP_LAYER_TYPE}_{EXP}.pt")
    torch.save(ModalityBuffer.inputs_emb_modalities, f"./attn_tensors/{MODALITY}_{DROP_LAYER_TYPE}_{EXP}_modalities.pt")
    for hook in all_hooks:
        hook.remove()

    outputs = tokenizer.decode(output_ids[0]).strip()
    print(outputs)
    
    plot_attn_vis(MODALITY, DROP_LAYER_TYPE, EXP, q=inp, ans=outputs, has_modality=HAS_MOD)
    

if __name__ == '__main__':
    main()