import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import transformers
import warnings

# FIXME: many warnings of loading non-meta weights to meta weight ##############
warnings.filterwarnings("ignore")

torch.set_warn_always(False)
transformers.utils.logging.set_verbosity_error()
transformers.set_seed(42)

def main():
    disable_torch_init()
    image = '/home/akane38/LLaVA/llava/serve/examples/extreme_ironing.jpg'
    image = Image.open(image)
    inp = 'What is unusual about this image?'
    model_path = 'liuhaotian/llava-v1.6-vicuna-7b'
    cache_dir = 'cache_dir'
    device = 'cuda:0'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    image_processor = processor
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    if type(image_tensor) is list:
        tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        tensor = image_tensor.to(model.device, dtype=torch.float16)

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
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0]).strip()
    print(outputs)

if __name__ == '__main__':
    main()