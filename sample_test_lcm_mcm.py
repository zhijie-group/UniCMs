import torch
from tqdm.auto import tqdm
import argparse
import functools
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union


from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import io


import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from huggingface_hub import create_repo
from packaging import version
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL, PixArtTransformer2DModel, DPMSolverMultistepScheduler
from diffusers import PixArtAlphaPipeline
import torch
from diffusers.image_processor import PixArtImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from tqdm.auto import tqdm
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy
logger = logging.get_logger(__name__)

torch.cuda.empty_cache()







def eval(ck_path,eval_len,inference_step,guidance_scale,device_num=0,is_lora=False,top_k=None,discription=""):
    import torch
    from tqdm.auto import tqdm  
    import json  
    import os
    from pathlib import Path
    from typing import List, Union


    from datasets import load_dataset
    from torchvision import transforms
    from PIL import Image
    import io


    import accelerate
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint
    import torchvision.transforms.functional as TF

    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration, set_seed

    from huggingface_hub import create_repo
    from packaging import version
    from torch.utils.data import default_collate
    from torchvision import transforms
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig


    import diffusers
    from diffusers import (
        AutoencoderKL,
        LCMScheduler,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.utils import check_min_version, is_wandb_available
    from diffusers.utils.import_utils import is_xformers_available
    from transformers import T5Tokenizer, T5EncoderModel
    from diffusers import AutoencoderKL, PixArtTransformer2DModel, DPMSolverMultistepScheduler
    from diffusers import PixArtAlphaPipeline
    import torch
    from diffusers.image_processor import PixArtImageProcessor
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
    from tqdm.auto import tqdm
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

    import html
    import inspect
    import re
    import urllib.parse as ul
    from typing import Callable, List, Optional, Tuple, Union
    from diffusers.image_processor import PixArtImageProcessor
    from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
    from diffusers.schedulers import DPMSolverMultistepScheduler
    from diffusers.utils import (
        BACKENDS_MAPPING,
        deprecate,
        is_bs4_available,
        is_ftfy_available,
        logging,
        replace_example_docstring,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    if is_bs4_available():
        from bs4 import BeautifulSoup

    if is_ftfy_available():
        import ftfy
    from diffusers.schedulers import LCMScheduler

    import sys
    os.chdir('/liymai24/sjtu/xck/Show-o')
    sys.path.append('/liymai24/sjtu/xck/Show-o')

    

    print(os.getcwd())
    print(sys.path)
    from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
    from training.utils import get_config, flatten_omega_conf, image_transform
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import torch
    import wandb
    from models import Showo, MAGVITv2, get_mask_chedule
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftModel
    from transformers import AutoTokenizer
    import torch.nn.functional as F
    from omegaconf import OmegaConf

    import clip
    import os
    import hpsv2
    import torch
    import ImageReward as RM
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from tqdm import tqdm  # 导入 tqdm 库
    def forward(
        model,
        input_ids,
        input_embeddings=None,
        attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        config=None,
        labels_mask_text=None,
        labels_mask_image=None,
        **kwargs,
    ):
        attention_mask = attention_mask.to(dtype=model.dtype)
        if input_embeddings is None:
            logits = model.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = model.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            raise NotImplementedError

        return logits

    def log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps))
    def gumbel_noise(t, generator=None):
        noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
        return -log(-log(noise))

    def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
        confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
        sorted_confidence = torch.sort(confidence, dim=-1).values
        cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
        masking = confidence < cut_off
        return masking

    def denoise(model,input_ids, input_ids_minus_lm_vocab_size,uncond_input_ids, uncond_prefix,attention_mask, config, generator, ratio, mask_token_id, noise_schedule,seq_len,temperature):
        if uncond_input_ids is not None and config.training.guidance_scale > 0:
            uncond_input_ids = torch.cat(
                [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            cond_logits, uncond_logits = forward(model,model_input, attention_mask=attention_mask).chunk(2)
            # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            # it seems that muse has different cfg setting
            logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
            logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
        else:
            logits = forward(model,input_ids, attention_mask=attention_mask)
            logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]

        # print(logits)

        probs = logits.softmax(dim=-1)
        sampled = probs.reshape(-1, logits.size(-1))
        
        if top_k!=None:

            topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)
            topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
            sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]
            sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])

        else:
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
        # sampled_ids = torch.argmax(sampled, dim=-1).view(*logits.shape[:-1])
        # print(sampled_ids)
        unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
        sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        
        # print(ratio)
        # print(noise_schedule)
        mask_ratio = noise_schedule(torch.tensor(ratio))
        # Computes the probabilities of each selected tokens.
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
        selected_probs = selected_probs.squeeze(-1)
        # print(selected_probs)
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
        
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.max(
            torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
        )
        # Adds noise for randomness
        temperature = temperature * (1.0 - ratio)
        # print(mask_len,temperature)
        # print(mask_len.shape)
        # print(selected_probs.shape)
        masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
        # print(masking.shape)
        # Masks tokens with lower confidence.
        input_ids[:, -(seq_len + 1):-1] = torch.where(masking, mask_token_id,
                                                        sampled_ids + config.model.showo.llm_vocab_size + 10)
        input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        # print(logits.shape,unknown_map.shape,masking.shape)
        # print(masking.dtype,masking.sum())
        # print(unknown_map.dtype,unknown_map.sum())
        return input_ids, input_ids_minus_lm_vocab_size, temperature,sampled_ids

    os.environ["WANDB_MODE"] = "offline"
    def get_vq_model_class(model_type):
        if model_type == "magvitv2":
            return MAGVITv2
        else:
            raise ValueError(f"model_type {model_type} not supported.")
        
    torch.cuda.empty_cache()

    config_file_path = '/liymai24/sjtu/xck/Show-o/configs/showo_demo_512x512.yaml'
    config = OmegaConf.load(config_file_path)
    config.mode='t2i'

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id
    
    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_t2i' + f'_{config.mode}',
        config=wandb_config,
    )
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    # model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    # model = Showo.from_pretrained(ck_path).to(device)
    # model.eval()
    if is_lora:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        input_dir=ck_path
        peft_config = LoraConfig.from_pretrained(input_dir)
        model = PeftModel.from_pretrained(model,input_dir, config=peft_config) 
        model = model.to(device)
        model.eval()
    else:
        input_dir=ck_path
        # input_dir=os.path.join(ck_path, "model")
        model = Showo.from_pretrained(input_dir).to(device)
        model.eval()


    dtype=torch.float16
    model = model.to(dtype=dtype)
    vq_model = vq_model.to(dtype=dtype)


    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = 1
    config.training.guidance_scale = guidance_scale
    config.training.generation_timesteps = inference_step

    mask_token_id = model.config.mask_token_id


    # 仅评测 style 为 photo 的部分
    with open("/liymai24/sjtu/xck/dataset/benchmark_photo.json", "r") as f:
        data = json.load(f)

    # 将纯列表转换为字典列表
    data_dict = [{"text": item} for item in data]
    from datasets import Dataset
    import pandas as pd
    # 创建数据集对象
    dataset = Dataset.from_pandas(pd.DataFrame(data_dict))
    prompts = dataset["text"][:eval_len]

    image_dir="/liymai24/sjtu/xck/image/test_lmcm_x_photo/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    text_dir = '/liymai24/sjtu/xck/image/prompt_lmcm_x_photo'
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    # 使用 tqdm 显示进度条
    # 使用 tqdm 显示进度条
    with tqdm(total=len(prompts), desc="Generating Images") as pbar:
        for idx, prompt in enumerate(prompts):
            text_filename = f'test_lmcm_x_photo_{idx}.txt'  # 与图片文件名一致
            output_text_path = os.path.join(text_dir, text_filename)
            with open(output_text_path, "w") as f:
                f.write(prompt)

            # 更新进度条
            pbar.update(1)
    print("prompt save done")
    # prompts=["A glossy, perfectly ripe ruby red apple sits on a pristine white plate placed on a rustic wooden table, adding warmth and texture to the scene. The apple’s skin gleams with a natural sheen, a small highlight catching the light to emphasize its freshness. A single small green leaf, crisp and vivid, is attached delicately to its stem, enhancing its fresh-picked charm. Behind the apple, soft morning light filters through a nearby window, casting gentle shadows and illuminating subtle details in the wood grain. In the blurred background, hints of a cozy kitchen are visible, with a vase of wildflowers adding a touch of color and natural beauty. The scene is inviting and appetizing, with the white plate and red apple standing out against the earthy, homely setting.",
    #          "A fluffy white cat with striking blue eyes lounges gracefully on a sunlit windowsill. The sunlight streaming through the window creates a warm, golden glow, casting soft shadows around the cat’s relaxed form. Its fur looks silky and slightly tousled, with hints of cream near the ears and paws. Outside the window, a vibrant garden filled with blooming flowers in shades of pink, yellow, and purple adds a burst of color to the scene. The cat’s gaze is calm yet curious, watching the flutter of a butterfly just beyond the glass. The entire scene is peaceful, cozy, and filled with a sense of gentle warmth.",
    #          "A close-up of an adorable golden retriever puppy, shown in a warm, slightly painterly style. The puppy has one eye closed in a playful wink, with its pink tongue just visible in a cheerful smile. It wears a red, white, and blue bandana that adds a touch of sparkle and texture. The background shows a vibrant, idyllic outdoor scene with lush green grass, a bright blue sky, fluffy white clouds, and soft sunbeams or a touch of lens flare, creating a joyful, lively aesthetic.",
    #          "A vibrant coral reef teems with life under crystal-clear turquoise waters, illuminated by sunlight filtering down from the surface. Brightly colored corals in shades of orange, pink, and purple create a stunning underwater landscape. Schools of tropical fish with vivid blue, yellow, and green scales dart through the water, while a graceful sea turtle glides by, adding a sense of peaceful motion. Shafts of sunlight dance on the sandy seabed, casting a mesmerizing pattern across the reef and highlighting its beauty.",
    #          "A lone whale gracefully rides a perfect wave against a vibrant sunset backdrop. The ocean reflects the warm hues of the sky, blending shades of pink, orange, and purple in a breathtaking display. The fish, silhouetted against the sunset, glides smoothly along the crest of the wave, its fins extended as if surfing the water. Gentle splashes and the flowing motion of the wave add dynamic energy to the scene, while the colorful reflections create a tranquil yet exhilarating atmosphere.",
    #          "A peaceful lake scene at the break of dawn, with still, mirror-like water reflecting the soft, golden light of sunrise. The lake is surrounded by towering pine trees and distant, mist-covered mountains, which fade softly into the horizon.  The sky transitions smoothly from shades of warm orange near the horizon to a calming blue above, creating a soothing and immersive atmosphere.",
    #         "A 3D render of a cute, smiling cactus character named Spike, standing in a sunny desert landscape. Spike has a round, plump body with short, stubby arms resembling cactus pads, covered in soft, harmless spines that give it a friendly and huggable appearance. Its face features big, sparkling eyes filled with curiosity and warmth, and a wide, cheerful smile that radiates friendliness. Spike has small pink flowers blooming atop its head, adding a touch of color and personality. The desert background features golden sand dunes with a few scattered rocks and small, distant cacti under a bright blue sky. The sunlight casts a warm glow, creating soft shadows and highlighting Spike’s vibrant green body, making it stand out against the warm desert tones. The overall scene is filled with warmth and positivity, inviting viewers to enjoy a delightful moment with this charming cactus character.",
    #         "This is a very clear and delicate portrait of a beautiful woman with long, flowing hair, scattered with delicate flowers. Her eyes are large and reflect an ethereal light, and her lips are gently curved into a gentle smile. She wears delicate, nature-inspired jewelry, with vines and leaves woven into the designs. The lighting is soft, casting a warm glow on her flawless skin, and the subtle patterns painted on her face add a mysterious quality. Her expression is calm and serene, evoking a sense of elegance and nobility.",
    #         "A photo-realistic image of a garden with pink and blue flowers. There are pink poppies in the foreground, with their petals gently curved. The background features purple cosmos flowers. The flowers have water droplets on their petals, which glisten in the natural light. The green leaves are lush and healthy. The background is blurred, with a few trees and buildings visible. The overall image has a high resolution and is hyper-realistic, as if taken by a skilled photographer.",
    #         "A stylized, whimsical cartoon tiger with large, expressive eyes and a playful smile, created in an abstract, vibrant art style. The tiger’s fur is adorned with bold, swirling patterns and geometric shapes in bright hues of orange, teal, and purple, making it visually striking. The background is a mix of abstract shapes and colors that complement the tiger's vibrant design, giving it a contemporary, artistic feel. The overall effect is lively and dynamic, as if the tiger is both friendly and majestic in a colorful, imaginative world.",
    #         ]
#     prompts = [
#     "A perfectly brewed cup of coffee sits on a rustic ceramic mug, its dark, rich liquid gleaming with steam rising in the cool morning air. The coffee is surrounded by a scattering of freshly baked croissants, golden and flaky, their layers visible as they catch the soft, warm light from a nearby window. In the background, a simple wooden table is adorned with an open book and a pair of reading glasses, creating a cozy, inviting atmosphere. The room’s soft, natural light highlights the textures of the croissants and the smoothness of the mug, evoking a sense of peace and comfort.",
#     "A vibrant field of sunflowers stretches as far as the eye can see, their golden faces turned toward the bright afternoon sun. The sky is a rich blue, dotted with a few fluffy clouds, creating a perfect contrast to the yellow and green of the field. In the foreground, a gentle breeze causes some of the sunflowers to sway, while a single bee hovers near one of the blossoms. The scene is peaceful, full of life, and radiates the warmth and beauty of summer.",
#     "A delicate porcelain teacup, filled with a steaming, fragrant green tea, sits atop a fine lace doily on an antique wooden table. A small, freshly picked sprig of mint is placed beside the cup, adding a pop of vibrant green. In the background, a window looks out to a garden filled with soft pink roses and lavender, their colors gently muted by the soft afternoon light. The scene feels tranquil, inviting relaxation and quiet contemplation.",
#     "A sleek, modern cityscape at twilight, with skyscrapers illuminated by the fading light of the day. The buildings reflect the vibrant hues of the sunset—shades of deep purple, fiery orange, and soft pinks—creating a breathtaking contrast with the cool blue of the evening sky. Streets below are bustling with the glow of traffic lights and the motion of people, giving the scene a dynamic, urban energy. The city feels alive, modern, and vibrant, with an edge of sophistication.",
#     "A tranquil winter scene in a snow-covered forest, with tall pines standing like silent sentinels against the pale, soft sky. Fresh snow gently drapes the branches, and delicate snowflakes continue to fall from above, adding to the peaceful serenity. In the distance, a small wooden cabin emits a soft, warm light through its windows, a contrast to the cool whites and blues of the snow. The scene feels calm and untouched, evoking a sense of solitude and quiet beauty.",
#     "A majestic eagle soars high above a rugged mountain range, its powerful wings outstretched against a clear blue sky. Below, the jagged peaks are dusted with snow, while deep valleys are blanketed in dark evergreen forests. Sunlight catches the eagle’s feathers, making them gleam with golden highlights. The eagle’s piercing gaze and graceful flight evoke a sense of freedom and strength, and the entire scene is awe-inspiring and grand.",
#     "A lush tropical rainforest, where the rich green foliage forms a dense canopy above a misty, winding stream. Brightly colored parrots and butterflies flutter among the trees, adding splashes of red, yellow, and blue to the vibrant greenery. The air is thick with moisture, and the sounds of water flowing and distant animal calls create an immersive, lively atmosphere. Sunlight filters through the thick leaves, casting dappled shadows on the ground and adding depth to the rich, layered landscape.",
#     "A vintage bicycle, its frame a soft pastel color, leans casually against an old brick wall, surrounded by wildflowers and climbing ivy. The sun is low in the sky, casting a golden glow across the scene, creating long shadows on the cobblestone street. The bike’s basket holds a bouquet of fresh flowers, while a gentle breeze stirs the leaves. The atmosphere is nostalgic, peaceful, and full of charm, evoking a sense of freedom and simple pleasures.",
#     "A close-up of a sunflower blooming in a garden, with its vibrant yellow petals unfurling toward the sun. The center of the flower is a perfect spiral of golden seeds, creating a natural pattern that draws the eye. In the background, the soft blur of more sunflowers and a few bees busy at work add life to the scene. The sunlight enhances the brightness of the petals, giving them a glowing, almost ethereal quality.",
#     "A serene moonlit night over a calm ocean, with the silver light of the moon reflecting on the dark water, creating a path of shimmering light that stretches toward the horizon. In the distance, a small sailboat drifts peacefully, its white sails illuminated by the soft glow of the moon. The sky is deep navy blue, with a scattering of stars adding to the stillness and majesty of the night. The scene is quiet, romantic, and full of mystery."
# ]
#     prompts = [
#     "A cozy fireplace crackles softly in a dimly lit living room, casting flickering shadows across the walls. A plush armchair, draped with a knitted blanket, sits invitingly beside the hearth. On the nearby coffee table, a steaming cup of hot cocoa with a cinnamon stick rests next to a plate of gingerbread cookies, their edges dusted with powdered sugar. The warm light from the fire dances across the room, illuminating the soft textures of the furniture and the twinkling glow of a small Christmas tree in the corner, creating an atmosphere of comfort and holiday cheer.",
#     "A secluded beach at sunrise, where soft pink and purple hues paint the sky, blending with the golden light reflecting off the gentle waves. A lone palm tree leans slightly toward the ocean, its fronds rustling in the breeze. The fine sand stretches into the distance, undisturbed, with only the footprints of a few seagulls marking the surface. The atmosphere is peaceful and serene, with the promise of a new day in the air.",
#     "A misty forest at dawn, where the fog clings to the towering trees and weaves around their trunks. The ground is covered with a blanket of fallen leaves, damp and dark from the morning dew. Faint beams of sunlight break through the mist, creating soft rays of light that illuminate pockets of wildflowers and ferns. The scene feels magical and untouched, filled with mystery and quiet wonder.",
#     "A futuristic cityscape at night, with sleek, glowing skyscrapers that tower over bustling streets below. Neon signs in bright colors light up the scene, casting colorful reflections on the wet pavement. Flying cars zip through the sky, their headlights adding streaks of light across the dark, starless night. The streets are crowded with people, their faces illuminated by the glow of digital billboards and street lamps. The atmosphere is vibrant, high-tech, and full of energy.",
#     "A serene garden pond, surrounded by lush greenery and vibrant blooms. The surface of the water is still, reflecting the blue sky above and the bright orange koi fish swimming lazily beneath. Lotus flowers float serenely on the surface, their petals slightly open in the sunlight. A small stone bridge arches gracefully over the pond, adding a peaceful and contemplative quality to the scene. The atmosphere is tranquil and idyllic, perfect for quiet reflection.",
#     "An old, cobblestone street in a quaint European village, with charming houses adorned with colorful shutters and hanging flower baskets. The street is bathed in the golden light of the late afternoon sun, casting long shadows and adding warmth to the weathered stone. A small café with outdoor seating spills out onto the sidewalk, where people enjoy coffee and pastries while chatting. The atmosphere is relaxed, timeless, and filled with the sounds of quiet conversation and clinking glasses.",
#     "A dramatic sunset over a vast desert landscape, where the sky is painted in fiery oranges and reds, contrasting with the deep browns and oranges of the sand dunes. A solitary cactus stands silhouetted against the horizon, its long shadows stretching across the desert floor. The air is dry and still, with only the occasional gust of wind sending sand drifting across the dunes. The scene feels both awe-inspiring and harsh, capturing the raw beauty of the desert.",
#     "A cozy reading nook by a window, with a plush armchair nestled beneath sheer curtains. A stack of books sits beside the chair, with an open novel resting on the seat. Soft, warm light filters through the window, casting a glow on the pages and creating a peaceful ambiance. A potted plant sits on the windowsill, its green leaves framing the view of a quiet street outside. The room is inviting and calm, the perfect place to lose oneself in a good story.",
#     # "A snowy mountain peak at twilight, where the last light of the day casts a pale glow over the crisp white snow. The jagged ridges of the mountain are bathed in soft pinks and purples, with the dark silhouette of a lone hiker standing near the summit, looking out over the vast expanse below. The sky above is deepening into a rich indigo, and the first stars begin to appear, twinkling against the fading light. The scene is majestic, tranquil, and filled with a sense of accomplishment.",
#     "A whimsical scene in a magical forest, where soft, glowing fireflies hover around the trees, casting a gentle light on the moss-covered ground. Towering mushrooms with brightly colored caps grow in clusters, and delicate, translucent butterflies flit through the air. The trees are tall and ancient, their trunks twisted in intricate patterns, and the air is filled with the faint scent of wildflowers. The scene is otherworldly, enchanting, and filled with the quiet hum of nature’s magic."
# ]
#     prompts = [
#     # "A foggy morning on a quiet lakeside, where the water is a mirror, perfectly reflecting the mist-covered trees and the soft gray sky above. A small rowboat is gently tethered to a wooden dock, its weathered planks worn by time. The air is crisp and still, the only sound being the occasional splash of a fish breaking the surface of the water. The atmosphere is serene and mysterious, evoking a sense of solitude and quiet reflection.",
#     # "A bustling café terrace in Paris, where patrons sip espresso and chat beneath the shade of large umbrellas. The cobblestone streets are alive with the soft hum of conversation and the sound of bicycle bells ringing in the distance. Nearby, a street artist sketches portraits, his easel surrounded by curious onlookers. In the background, the Eiffel Tower looms gracefully, its iron lattice framed by the soft afternoon light. The scene is vibrant, full of life, and unmistakably romantic.",
#     # "A vibrant autumn forest, where the trees are ablaze with colors of deep red, orange, and yellow. A narrow trail winds through the woods, leading deeper into the dense foliage. The ground is covered in a thick layer of fallen leaves, crunching beneath each step. Sunlight filters through the canopy, casting dappled light on the forest floor. The atmosphere is warm, earthy, and filled with the scent of pine and damp leaves, evoking the richness of the fall season.",
#     # "A charming coastal village at dusk, where the soft glow of lanterns illuminates narrow stone streets. The air is filled with the salty scent of the sea and the sound of gentle waves crashing against the rocks. Small boats bob in the harbor, their masts swaying gently in the evening breeze. The buildings are painted in soft pastels, with windows framed by colorful shutters. The atmosphere is peaceful and inviting, evoking a sense of nostalgia and the quiet beauty of coastal living.",
#     # "A snowy mountain pass, where the towering peaks are bathed in the soft, golden light of the setting sun. The air is thin and crisp, with a sharp bite to it. A lone hiker, bundled in layers, stands at the summit, gazing out over the vast expanse of snow-covered hills and valleys stretching below. The landscape is both harsh and breathtaking, with jagged cliffs and glaciers visible in the distance. The scene feels both exhilarating and humbling, capturing the grandeur of nature in its rawest form.",
#     # "A bustling street in Tokyo at night, where neon signs in bright colors reflect off the slick pavement. The streets are alive with people, some rushing by in business attire, while others wander through the glowing arcades and shops. The scent of freshly grilled yakitori wafts through the air, and the soft sound of J-pop music spills from the open doors of nearby stores. The energy of the city is palpable, blending tradition with modernity in a vibrant, fast-paced environment.",
#     # "A peaceful meadow at sunrise, where wildflowers bloom in clusters of purple, yellow, and white, dotting the soft green grass. A narrow stream winds its way through the meadow, its water sparkling in the early morning light. A lone deer grazes in the distance, its ears twitching at every sound. The air is cool and fresh, with a faint scent of dew and earth. The scene is calm and idyllic, capturing the quiet beauty of the natural world at the start of a new day.",
#     # "A traditional Japanese garden, where meticulously trimmed bonsai trees sit in perfect harmony with smooth stones and shallow koi ponds. The path is lined with bamboo fences, leading past delicate cherry blossoms in full bloom. A small wooden bridge arches over a pond filled with brightly colored fish. The serene atmosphere is enhanced by the sound of wind chimes gently ringing in the breeze, creating a sense of balance, peace, and timeless beauty.",
#     "A sun-dappled vineyard at the peak of summer, where rows of grapevines stretch toward the horizon, heavy with ripening fruit. The air is warm and fragrant with the scent of fresh grapes and the earth. In the distance, a stone farmhouse with a red-tiled roof stands against a backdrop of rolling hills. The sky is a deep blue, with only a few wispy clouds. The scene is filled with the richness of the harvest season, evoking a sense of abundance and the beauty of rural life.",
#     # "A vintage bookstore tucked in a quiet alley, where shelves of old, leather-bound books rise to the ceiling, each volume with its own story to tell. The air is filled with the musky scent of aged paper, and soft light filters through the dusty windows, casting a warm glow on the worn wooden floors. A small armchair sits near a crackling fireplace, inviting visitors to sit and lose themselves in a good book. The atmosphere is cozy, intimate, and filled with the quiet joy of discovering forgotten treasures."
# ]
    # prompts = [
    # "A quiet country road at dawn, where the soft glow of the rising sun bathes the landscape in hues of pink, orange, and gold. The road is flanked by ancient oaks whose branches stretch wide, their leaves rustling in the gentle morning breeze. In the distance, a small farmhouse with a red roof is partially obscured by mist, adding to the peaceful, dreamlike atmosphere. The air is fresh and cool, filled with the scent of grass and earth, evoking a sense of calm and serenity.",
    # "A vibrant carnival at night, with brightly colored lights flashing in all directions. The ferris wheel spins slowly in the background, its glowing lights contrasting against the dark sky. Stalls filled with cotton candy, balloons, and carnival games line the pathway, where children laugh and excitedly chase each other. The atmosphere is filled with joy, energy, and nostalgia, as the sounds of music and laughter blend with the sweet scent of popcorn and candy apples.",
    # "A secluded beach during a summer evening, where the sky is painted in warm shades of pink and purple as the sun sets over the horizon. The ocean is calm, with gentle waves lapping at the shore, reflecting the colors of the sky. A hammock is strung between two palm trees, swaying lazily in the breeze. The air is warm and salty, and the faint scent of tropical flowers drifts on the wind. The scene is relaxed and tranquil, perfect for a quiet retreat by the sea.",
    # "A cozy breakfast nook in a sunlit kitchen, where a steaming cup of freshly brewed coffee sits on a wooden table next to a plate of buttery croissants. The light from the window creates a warm, golden glow, casting soft shadows across the room. Fresh fruit, vibrant with color, sits in a bowl on the counter. The scent of fresh bread and citrus fills the air, creating a welcoming atmosphere that feels both comforting and energizing.",
    # "A bustling street market in Morocco, where colorful textiles hang from every stall, and the scent of spices and fresh herbs fills the air. People move through the narrow lanes, bargaining for vibrant handwoven rugs, intricately carved lanterns, and fragrant oils. The warm sun casts a golden hue over the scene, while the distant call to prayer and the rhythmic tapping of a metal drum add to the lively atmosphere. The entire scene feels alive, full of culture, history, and the rich flavors of Morocco.",
    # "A quiet mountaintop at sunrise, where the first light of the day spills over the jagged peaks, casting long shadows across the rocky terrain. The air is thin and crisp, with the scent of pine and fresh snow. In the distance, a small wooden cabin stands against the rugged landscape, its chimney releasing a thin plume of smoke. The stillness of the morning is interrupted only by the occasional call of an eagle soaring overhead. The atmosphere is peaceful, awe-inspiring, and full of the majesty of nature.",
    # "A tranquil garden at dusk, where the last rays of sunlight filter through the leaves of tall, swaying bamboo. The gentle sound of a nearby fountain adds to the calming atmosphere, while delicate lanterns hanging from trees cast a soft glow across the path. A small wooden bench sits under an arch of climbing ivy, inviting a moment of quiet contemplation. The air is cool and filled with the scent of jasmine and evening blooms, creating a sense of peace and introspection.",
    # "A lively jazz club in New Orleans, where the smooth sounds of a saxophone fill the air, blending with the laughter and chatter of the crowd. The dim lighting reflects off polished wooden tables, and the clink of glasses adds to the vibrant rhythm of the night. On stage, a jazz band plays passionately, their energy infectious. The atmosphere is electric, with a mix of excitement, nostalgia, and the smooth, soulful charm of the city.",
    # "A sprawling lavender field in Provence, where rows of vibrant purple flowers stretch to the horizon, their scent wafting on the warm summer breeze. The sky above is a soft blue, dotted with a few clouds, and the sun shines down, creating a patchwork of light and shadow on the ground. In the distance, a small stone farmhouse with a red-tiled roof stands against the lavender landscape. The scene is peaceful, fragrant, and infused with the natural beauty of the French countryside.",
    # "A vintage train station at twilight, where the soft glow of the station lights casts long shadows on the platform. The air is cool and still, and the sound of distant train whistles can be heard on the tracks. A single train, its dark green carriages gleaming in the soft light, waits at the station, ready to depart. The scene evokes a sense of nostalgia, with the quiet anticipation of travel and the charm of a bygone era of train journeys."
# ]

    # prompts=[
    #     "A snowy mountain peak at twilight, where the last light of the day casts a pale glow over the crisp white snow. The jagged ridges of the mountain are bathed in soft pinks and purples, with the dark silhouette of a lone hiker standing near the summit, looking out over the vast expanse below. The sky above is deepening into a rich indigo, and the first stars begin to appear, twinkling against the fading light. The scene is majestic, tranquil, and filled with a sense of accomplishment.",
    #     "A sun-dappled vineyard at the peak of summer, where rows of grapevines stretch toward the horizon, heavy with ripening fruit. The air is warm and fragrant with the scent of fresh grapes and the earth. In the distance, a stone farmhouse with a red-tiled roof stands against a backdrop of rolling hills. The sky is a deep blue, with only a few wispy clouds. The scene is filled with the richness of the harvest season, evoking a sense of abundance and the beauty of rural life.",
    #     "A secluded beach during a summer evening, where the sky is painted in warm shades of pink and purple as the sun sets over the horizon. The ocean is calm, with gentle waves lapping at the shore, reflecting the colors of the sky. A hammock is strung between two palm trees, swaying lazily in the breeze. The air is warm and salty, and the faint scent of tropical flowers drifts on the wind. The scene is relaxed and tranquil, perfect for a quiet retreat by the sea.",
    #     "A dolphin leaps in the ocean waves against a bright blue and pink background.",
    #     "A cozy winter cabin in a snowy forest during a bright, sunny day, with sunlight reflecting off the snow and a clear blue sky.",
    # ]
#     prompts = [
#     "A delicate porcelain teapot sitting on a polished wooden table, its surface intricately decorated with blue and white floral patterns. The teapot is filled with freshly brewed green tea, steam gently rising from the spout. Nearby, a small porcelain cup rests on a saucer, its edges decorated with a gold rim. The soft light of the afternoon sun casts a warm glow on the tea set, highlighting the delicate details and creating an atmosphere of quiet elegance and relaxation.",
#     "An antique typewriter, its keys worn from years of use, sits atop a cluttered desk in a dimly lit study. Papers with handwritten notes and half-finished stories are scattered around it. The soft click-clack of the keys seems to echo in the quiet room, where the smell of old books and ink fills the air. The typewriter, though aged, holds the promise of creativity and nostalgia, evoking the romance of a bygone era of writing.",
#     "A hand-painted ceramic bowl, rich with vibrant hues of turquoise, orange, and gold, rests on a rustic wooden table. The bowl is filled with ripe fruits—shiny red apples, golden pears, and plump grapes—arranged in an artful display. The light from a nearby window falls across the table, casting intricate shadows that dance across the polished surface of the bowl. The scene feels abundant, colorful, and inviting, capturing the essence of nature’s harvest.",
#     "A worn leather-bound journal with brass clasps sits open on a wooden desk, its pages yellowed with age. The pen next to it is a fountain pen, its nib inked and ready to continue a handwritten story. The room is bathed in the soft glow of an antique lamp, and the air smells of old paper and ink. The atmosphere is one of quiet reflection and creative flow, evoking a sense of timeless storytelling.",
#     "A single red rose, freshly picked, lies gently on a white lace handkerchief. The velvety petals of the rose are dewdrop-streaked, and its stem is wrapped with a simple green ribbon. The soft lighting from a nearby window casts a delicate shadow, enhancing the natural beauty of the flower. The scene is both elegant and intimate, filled with the timeless symbolism of love and beauty.",
#     "A shiny, vintage bicycle resting against a brick wall in an urban alley. The bike is painted in a bright shade of blue, with a leather seat and chrome handlebars that reflect the soft glow of the streetlights. Nearby, a small potted plant adds a touch of greenery to the urban scene. The quiet hum of the city outside contrasts with the calm, nostalgic atmosphere of the bike, evoking a sense of adventure and freedom.",
#     "A silver pocket watch with intricate engravings on the back sits open on a polished mahogany desk. The watch's ticking is soft and steady, marking the passage of time in the quiet room. The light from a nearby lamp casts a warm, golden glow on the watch’s reflective surface, highlighting its elegance and craftsmanship. The scene feels both timeless and serene, as if caught in a moment of reflection.",
#     "A small, hand-carved wooden figurine of a dancer, with flowing robes and outstretched arms, stands atop a simple wooden pedestal. The figurine’s graceful pose seems to capture the motion of dance in a moment frozen in time. The wood is rich and dark, its grains visible and smooth to the touch. The light falls gently on the figure, creating delicate shadows that add depth and elegance to the scene.",
#     "A weathered, leather backpack rests on a moss-covered stone in a quiet forest clearing. The backpack, worn but sturdy, is packed with camping gear—a rolled-up tent, a blanket, and a small thermos. The sunlight filters through the trees above, casting dappled light onto the backpack and the surrounding forest floor. The scene feels peaceful and exploratory, filled with a sense of adventure and the promise of the great outdoors.",
#     "A golden retriever puppy, its fur soft and fluffy, sits in a patch of sunlight on a grassy lawn. The puppy’s big, expressive eyes are filled with innocence and curiosity, and its tail wags excitedly as it spots a butterfly fluttering nearby. The light dapples through the trees above, creating a playful and joyful atmosphere, filled with warmth and the pure joy of a young puppy discovering the world around it."
# ]
#     prompts = [
#     # "A bright yellow sunflower in full bloom stands tall against a clear blue sky. The petals are wide and vibrant, with a soft gradient of orange at the tips. The center of the flower is a deep brown, with a subtle texture that contrasts beautifully with the smooth, green leaves at the base. The scene is simple, cheerful, and radiates warmth and energy, perfect for a fresh and inviting display.",
#     "A fresh cup of steaming coffee with a soft swirl of milk sits on a white saucer. The cup is plain white, and the coffee’s surface is glossy and dark, with a small heart-shaped pattern formed by the milk. A single cinnamon stick rests beside the cup, and a sprig of mint adds a touch of green to the scene. The light falls gently on the cup, creating a cozy, inviting atmosphere that's perfect for a minimalistic yet warm display.",
#     "A single green apple with a small red blush sits on a simple white plate. The apple’s skin is smooth and shiny, reflecting the light from above. The plate is plain with a soft, smooth surface, and there is a single green leaf attached to the apple’s stem. The background is softly blurred, with hints of light creating a clean, crisp, and refreshing feel, perfect for a bright and inviting visual.",
#     "A small glass jar of honey sits on a wooden table, with a golden honeycomb pattern visible inside. A wooden honey dipper rests beside the jar, with a small drizzle of honey glistening from its tip. The jar’s light amber color contrasts with the soft, natural wood tones of the table. The entire scene feels simple, fresh, and natural, creating a sense of calm and sweetness.",
#     # "A minimalist cactus in a white ceramic pot sits on a windowsill bathed in natural light. The cactus is small, with plump, green pads that contrast against the smooth, glossy surface of the pot. The soft light from the window highlights the clean lines of the pot and the simple beauty of the cactus, evoking a feeling of calm, freshness, and easy elegance.",
#     "A bunch of ripe bananas with bright yellow skin rests on a white plate. The bananas are slightly curved, their texture smooth and glossy. The plate is plain, drawing focus to the fruit, while the background is a soft, light pastel color that complements the vibrant yellow of the bananas. The scene feels fresh, simple, and full of life, perfect for a bright, cheerful visual.",
#     "A small, potted succulent with round, fleshy leaves sits on a sleek white shelf. The succulent’s leaves are a soft, dusty green with subtle hints of pink on the edges. The pot is simple, smooth, and white, creating a contrast with the textured leaves. The scene is clean, minimal, and modern, with soft light falling gently on the plant, giving it a fresh and serene vibe.",
#     "A clear glass bottle filled with vibrant pink lemonade sits on a wooden table, with a slice of lemon perched on the rim of the bottle. The pink liquid glistens in the light, and a sprig of mint adds a touch of green. The background is blurred with soft, warm tones, creating a light, refreshing, and inviting atmosphere that is perfect for a clean and summery display.",
#     "A simple white bowl filled with fresh strawberries sits on a marble countertop. The strawberries are plump and red, their surfaces dotted with tiny seeds. Some of the berries are cut in half, revealing their juicy interior. The scene is bright and fresh, with soft lighting highlighting the natural beauty of the fruit, making it look both inviting and delicious.",
#     "A pale blue ceramic mug with a minimalist design sits on a wooden table next to a small plate of oatmeal cookies. The mug is simple, with a curved handle, and the oatmeal cookies are golden-brown, with chunks of chocolate and raisins peeking out. The light from a nearby window softly illuminates the scene, creating a cozy, comforting atmosphere that feels perfect for a warm, inviting display."
# ]
#     prompts = [
#     # "A single slice of lemon rests on a white marble countertop, its bright yellow color vibrant against the smooth surface. A few drops of juice glisten on the edge of the slice, and a fresh mint leaf is tucked beside it. The scene is clean and fresh, with soft light highlighting the citrusy freshness and a touch of natural green, evoking a refreshing, zesty vibe.",
#     "A small glass jar filled with lavender buds sits on a wooden table. The jar is simple and clear, allowing the soft purple hue of the lavender to stand out. A few sprigs of lavender are placed next to the jar, adding to the calming, natural atmosphere. The background is softly blurred with a warm, golden glow, creating a serene and peaceful mood.",
#     "A bowl of ripe, shiny cherries sits on a pale blue plate. The cherries are plump and deeply red, with a single leaf attached to one of the stems. The light falls softly on the surface, making the fruit appear fresh and glossy. The simple, clean setup and the bright pop of red create a vibrant, cheerful, and inviting atmosphere.",
#     "A tiny glass vase holds a single white daisy, its petals open wide to the soft sunlight streaming through the window. The vase is clear and minimalistic, allowing the delicate flower to take center stage. A few green leaves rest gently against the vase, completing the simple, elegant scene that radiates freshness and tranquility.",
#     "A stack of fluffy pancakes drizzled with golden maple syrup sits on a plain white plate. A dollop of butter rests on top, slowly melting into the warm pancakes. A few fresh berries, red and blue, are scattered around the plate, adding a pop of color. The light in the scene is soft, creating a warm and inviting feeling of comfort and sweetness.",
#     "A small, potted fern sits on a windowsill, its bright green fronds reaching toward the light. The pot is simple and white, contrasting with the rich green of the leaves. The sunlight filters through the window, casting soft shadows on the table below. The scene feels fresh, calming, and full of natural life, perfect for a clean and serene visual.",
#     "A freshly peeled orange sits on a light wood cutting board, its juicy segments exposed. The vibrant orange of the fruit contrasts beautifully against the soft natural wood tones. A couple of peel curls rest beside the orange, adding texture to the scene. The background is light and airy, evoking a feeling of freshness and vitality.",
#     "A vibrant green cactus in a simple terracotta pot sits on a light wooden shelf. The cactus is small, with perfectly arranged, spiky pads. The soft light from a nearby window creates gentle shadows, enhancing the texture of the plant and the smooth surface of the pot. The scene is minimal, fresh, and slightly playful, with a touch of natural charm.",
#     # "A bright pink cupcake with swirled frosting sits on a white porcelain plate. The frosting is perfectly piped, with a light dusting of sprinkles on top. The cupcake’s wrapper is a soft pastel color, complementing the bright frosting. The light, airy scene feels sweet and inviting, with a playful, colorful touch that makes it perfect for a cheerful display.",
#     "A small white mug filled with a hot latte sits on a simple wooden table. The top of the latte is decorated with a beautiful leaf-shaped pattern, created by the milk foam. The mug is plain, with a smooth, rounded shape, and the light from a nearby window casts soft shadows on the table. The atmosphere feels warm, cozy, and comforting, perfect for a simple yet inviting visual."
# ]
#     prompts = [
#     "A small glass jar filled with vibrant orange carrot sticks sits on a wooden table. The carrots are fresh and crisp, their vibrant color standing out against the soft, natural wood of the table. A few green sprigs of parsley are tucked between the carrots, adding a touch of green. The light from a nearby window creates a fresh and healthy vibe, perfect for a clean and refreshing display.",
#     "A perfectly sliced avocado rests on a light wooden cutting board. The smooth green flesh of the avocado contrasts with the soft brown of the skin. A slice of lime sits next to it, adding a pop of citrusy green. The background is soft and blurred, with natural light highlighting the fresh, creamy texture of the avocado.",
#     "A bright red bell pepper sits on a simple white plate, its smooth skin gleaming in the soft light. The pepper is cut in half, revealing its bright yellow interior and seeds. The simplicity of the plate and the pepper's rich color create a vibrant and fresh atmosphere, evoking a sense of health and vitality.",
#     # "A soft blue sky with a few fluffy clouds is reflected in a calm lake. A single white swan glides gracefully across the water, its feathers pristine and smooth. The light from the sun creates a shimmering effect on the surface of the water, while the simplicity of the scene feels serene and peaceful, perfect for a bright and calming visual.",
#     "A small wooden tray holds a cup of steaming green tea, surrounded by delicate jasmine flowers. The tea’s surface shimmers with a soft green hue, and a thin trail of steam rises gently into the air. The background is warm and soft, with light spilling across the tray, creating a peaceful, cozy, and inviting atmosphere.",
#     "A single pink tulip stands tall in a clear glass vase. The tulip’s petals are soft and smooth, with subtle gradients of pink and white. The vase is simple, allowing the delicate flower to stand out. The light coming from a nearby window highlights the soft curves of the flower, creating a clean and elegant scene.",
#     "A small bowl of freshly picked blueberries sits on a white ceramic plate. The berries are plump and juicy, their dark blue color contrasting with the white plate. A few leaves from a nearby plant are placed alongside the bowl, adding a fresh, green touch. The soft light in the scene creates a natural, fresh, and inviting feeling.",
#     # "A single bright yellow daffodil sits in a small glass vase on a windowsill. The flower’s vibrant petals catch the light, creating a cheerful and warm atmosphere. The vase is clear, allowing the simplicity of the flower and the light to take center stage. The background is softly blurred, with light spilling across the vase and the table beneath it.",
#     "A freshly baked loaf of bread rests on a soft cloth napkin, its golden crust still warm. The bread has a rustic, inviting look, with slight cracks on the surface. A small wooden butter knife rests beside it, ready to spread a dollop of creamy butter. The scene is warm and comforting, filled with the simple pleasure of homemade food.",
#     "A pale pink ceramic plate holds a small assortment of ripe strawberries, their surfaces dotted with tiny seeds. The strawberries are fresh and glossy, with a few leaves still attached to their stems. The simplicity of the plate and the vibrant red of the fruit create a lively and fresh scene, perfect for a bright and appetizing visual."
# ]
    # prompts = [
    # "A vibrant red apple with a shiny skin sits on a rustic wooden table, its rich color contrasting beautifully against the soft, neutral tones of the background. A soft light filters through a nearby window, casting gentle shadows across the apple and highlighting the wood grain beneath. In the distance, a few potted plants with lush green leaves are softly blurred, adding depth and a touch of natural life to the scene.",
    # "A perfectly frosted cupcake with pink icing and colorful sprinkles sits on a pale blue plate. The cupcake's soft texture and bright colors are the focal point, while the background features a subtle arrangement of soft-focus flowers in pastel hues. The light from the window gently illuminates the scene, creating a warm and inviting atmosphere that feels cheerful yet relaxed.",
    # # "A single yellow sunflower in a clear glass vase stands proudly on a windowsill, the bright yellow petals catching the sunlight. The background is a softly blurred garden with dappled light filtering through leaves, creating a sense of tranquility and depth. The warm light adds a soft glow to the scene, emphasizing the natural beauty of the flower while giving the background a lively, yet calm feel.",
    # "A small, vibrant cactus sits in a simple terracotta pot on a wooden shelf. The cactus is a rich green, and its sharp needles create a striking contrast against the soft, neutral tones of the background. Behind it, a shelf lined with small decorative items and soft plants adds texture and interest without overpowering the cactus, creating a balanced and inviting scene.",
    # "A steaming mug of hot cocoa, topped with whipped cream and a dusting of cocoa powder, sits on a cozy knitted coaster. The warm brown of the cocoa contrasts with the bright white of the cream. The background features a soft, blurred scene of a wooden mantel adorned with fairy lights, creating a warm and magical ambiance that enhances the comforting mood of the cocoa.",
    # # "A ripe peach with a soft, fuzzy skin rests on a white ceramic plate. The peach's rich orange and red hues pop against the simplicity of the plate. In the background, a wooden countertop is slightly out of focus, with a soft, leafy plant and a few kitchen utensils adding texture to the scene. The light from a nearby window creates soft highlights on the peach, enhancing its freshness and juiciness.",
    # "A fresh slice of watermelon with vibrant red flesh and a scattering of black seeds sits on a plain white plate. The rich color of the fruit contrasts with the clean background of a rustic kitchen counter, where a few ripe fruits and a small jug of water sit in soft focus. The overall scene feels refreshing, bright, and full of life, inviting the viewer to indulge in summer’s sweetness.",
    # "A freshly baked loaf of bread rests on a wooden board, its golden crust cracked open to reveal the soft, fluffy interior. The scene is lit warmly, with the background featuring a softly blurred kitchen counter with a few baking ingredients and a jar of honey, adding context to the rustic charm of the loaf without distracting from the bread itself.",
    # "A small glass jar filled with vibrant green mint leaves sits on a wooden table. The mint's leaves are fresh and crisp, their green color popping against the neutral background. The table is subtly decorated with a few scattered ingredients—such as lemon wedges and a small honey pot—hinting at a refreshing drink or recipe, giving the scene a layered, inviting feel.",
    # "A bright orange tulip in a clear glass vase stands against a soft-focus background of green foliage. The tulip’s petals are bold and vivid, catching the light and creating a striking contrast with the blurred leaves and trees in the background. The scene is simple yet full of life, with the flower's vibrant color serving as the focal point, while the surrounding greenery adds depth and warmth."
# ]

#     prompts = [
#     # "A cozy outdoor patio with a wooden table set for tea. A teapot, a cup of tea, and a small plate of cookies sit on the table. The background shows a gentle, lush garden with trees swaying in the breeze and soft light filtering through the leaves. A few potted plants line the edge of the patio, creating a warm and welcoming atmosphere. The scene feels peaceful and inviting, perfect for a relaxing afternoon.",
#     "A spacious kitchen with sunlight streaming through the windows, casting warm light across the counter. On the counter sits a bowl of fresh fruit—bananas, apples, and oranges—surrounded by small potted plants. In the background, a rustic wooden shelf holds neatly arranged spices and utensils, adding depth and texture to the scene while keeping the focus on the fresh produce.",
#     "A calm lake surrounded by rolling hills, with a single small boat gently drifting on the water. The boat is empty, its wooden frame reflecting on the glassy surface of the lake. In the distance, a few birds fly across the soft orange and pink hues of the setting sun. The scene is peaceful and expansive, with the gentle ripples of the water creating a soothing sense of serenity.",
#     # "A bright and airy living room with large windows letting in natural light. A plush couch sits near the window, with a couple of pillows in pastel shades. The coffee table has a small vase of flowers, and a soft rug adds texture to the wooden floor. In the background, a few plants and bookshelves add life to the space, creating a cozy, inviting atmosphere with a hint of modern charm.",
#     # "A wide meadow filled with vibrant wildflowers in every color, stretching toward a distant mountain range. A dirt path winds through the flowers, leading the eye toward the distant peaks. The sky is bright and clear, with a few fluffy clouds scattered across the horizon. The expansive scene feels peaceful and full of natural beauty, with the colors of the flowers contrasting beautifully with the green of the grass.",
#     "A rustic dining room with a wooden table set for dinner. The table is adorned with a simple white tablecloth, and on it sits a bowl of fresh salad, a pitcher of water, and a basket of warm bread. The background features a warm-toned fireplace with a flickering fire, adding a touch of cozy ambiance to the scene. Soft lighting from a hanging chandelier casts a gentle glow on the table, creating a welcoming and relaxed atmosphere.",
#     "A scenic coastal view from a cliff, overlooking the sparkling ocean below. The cliffs are dotted with wild grasses and a few small shrubs, adding texture and color to the landscape. In the distance, a lighthouse stands tall, its white and red stripes visible against the blue of the sky and sea. The vastness of the ocean and the bright sky create a sense of freedom and tranquility.",
#     "A sun-dappled park with a walking path leading through rows of trees. Benches are placed along the path, inviting passersby to sit and enjoy the peaceful surroundings. In the background, children play near a fountain, and a small café with outdoor seating can be seen just beyond the trees. The scene feels bright, lively, and full of warmth, with the soft light filtering through the canopy of leaves above.",
#     "A cozy bedroom with a large bed covered in soft white linens, accented by plush pillows in gentle pastel colors. A wooden nightstand beside the bed holds a small vase of flowers and a book. The background shows a gently lit window with sheer curtains, and a few indoor plants add a touch of greenery to the room. The overall atmosphere is calming, serene, and inviting, perfect for a peaceful night’s rest.",
#     "A spacious balcony overlooking a cityscape, with plants hanging from the railing and a small table set with a vase of fresh flowers. The background reveals a bustling city below, with the soft glow of the evening sun casting long shadows across the rooftops. The scene is both lively and relaxing, offering a sense of openness and connection to the city while still feeling private and peaceful."
# ]

    # prompts=[
    #     "A bright yellow sunflower in full bloom stands tall against a clear blue sky. The petals are wide and vibrant, with a soft gradient of orange at the tips. The center of the flower is a deep brown, with a subtle texture that contrasts beautifully with the smooth, green leaves at the base. The scene is simple, cheerful, and radiates warmth and energy, perfect for a fresh and inviting display.",
    #     "A minimalist cactus in a white ceramic pot sits on a windowsill bathed in natural light. The cactus is small, with plump, green pads that contrast against the smooth, glossy surface of the pot. The soft light from the window highlights the clean lines of the pot and the simple beauty of the cactus, evoking a feeling of calm, freshness, and easy elegance.",
    #     "A single slice of lemon rests on a white marble countertop, its bright yellow color vibrant against the smooth surface. A few drops of juice glisten on the edge of the slice, and a fresh mint leaf is tucked beside it. The scene is clean and fresh, with soft light highlighting the citrusy freshness and a touch of natural green, evoking a refreshing, zesty vibe.",
    #     # "A bright pink cupcake with swirled frosting sits on a white porcelain plate. The frosting is perfectly piped, with a light dusting of sprinkles on top. The cupcake’s wrapper is a soft pastel color, complementing the bright frosting. The light, airy scene feels sweet and inviting, with a playful, colorful touch that makes it perfect for a cheerful display.",
    #     "A soft blue sky with a few fluffy clouds is reflected in a calm lake. A single white swan glides gracefully across the water, its feathers pristine and smooth. The light from the sun creates a shimmering effect on the surface of the water, while the simplicity of the scene feels serene and peaceful, perfect for a bright and calming visual.",
    #     "A single bright yellow daffodil sits in a small glass vase on a windowsill. The flower’s vibrant petals catch the light, creating a cheerful and warm atmosphere. The vase is clear, allowing the simplicity of the flower and the light to take center stage. The background is softly blurred, with light spilling across the vase and the table beneath it.",
    #     "A single yellow sunflower in a clear glass vase stands proudly on a windowsill, the bright yellow petals catching the sunlight. The background is a softly blurred garden with dappled light filtering through leaves, creating a sense of tranquility and depth. The warm light adds a soft glow to the scene, emphasizing the natural beauty of the flower while giving the background a lively, yet calm feel.",
    #     "A ripe peach with a soft, fuzzy skin rests on a white ceramic plate. The peach's rich orange and red hues pop against the simplicity of the plate. In the background, a wooden countertop is slightly out of focus, with a soft, leafy plant and a few kitchen utensils adding texture to the scene. The light from a nearby window creates soft highlights on the peach, enhancing its freshness and juiciness.",
    #     "A cozy outdoor patio with a wooden table set for tea. A teapot, a cup of tea, and a small plate of cookies sit on the table. The background shows a gentle, lush garden with trees swaying in the breeze and soft light filtering through the leaves. A few potted plants line the edge of the patio, creating a warm and welcoming atmosphere. The scene feels peaceful and inviting, perfect for a relaxing afternoon.",
    #     "A bright and airy living room with large windows letting in natural light. A plush couch sits near the window, with a couple of pillows in pastel shades. The coffee table has a small vase of flowers, and a soft rug adds texture to the wooden floor. In the background, a few plants and bookshelves add life to the space, creating a cozy, inviting atmosphere with a hint of modern charm.",
    #     "A wide meadow filled with vibrant wildflowers in every color, stretching toward a distant mountain range. A dirt path winds through the flowers, leading the eye toward the distant peaks. The sky is bright and clear, with a few fluffy clouds scattered across the horizon. The expansive scene feels peaceful and full of natural beauty, with the colors of the flowers contrasting beautifully with the green of the grass.",
    #     # "A charming café terrace with small round tables, each adorned with a tiny vase of fresh flowers. The background shows a bustling street with pedestrians walking by and colorful storefronts lining the street. The sun casts a warm glow over the scene, highlighting the café's outdoor seating area and giving the atmosphere a cozy, welcoming vibe. A gentle breeze moves the leaves of nearby plants, adding life to the setting.",
    # "A bright and spacious living room with a wide window that overlooks a sunny garden. The couch is plush, with soft cushions in shades of green and beige. A wooden coffee table in the center holds a small potted plant and a few magazines. In the background, the garden outside is alive with blooming flowers and greenery, and the sunlight filters through the trees, casting gentle shadows on the floor.",
    # "A tranquil mountain retreat, where a wooden deck overlooks a serene lake surrounded by tall pine trees. The deck is furnished with a couple of lounge chairs, a small table, and a few plants. The lake reflects the blue sky and the surrounding mountains, creating a peaceful and expansive view. The scene is quiet and expansive, evoking a sense of calm and connection with nature.",
    # # "A rooftop garden in the city, with potted plants and small trees providing a green oasis amidst the urban landscape. A comfortable lounge chair sits next to a small table with a glass of lemonade, offering a perfect spot to relax. The city skyline is visible in the background, with tall buildings and a bright blue sky. The contrast between the lush greenery and the urban backdrop creates a unique, calming escape in the heart of the city."

    # ]

#     prompts = [
#     "A dreamy forest with glowing mushrooms and wisps of soft light floating through the trees. The atmosphere is ethereal, with mist swirling around the base of the trees, and soft pastel hues of pink, lavender, and blue fill the air. A small stream runs through the scene, sparkling like liquid silver under the faint glow of a crescent moon. The whole scene feels like stepping into a magical dream, full of mystery and wonder.",
#     "A mystical castle floating among the clouds, its spires reaching towards the soft, golden sky at sunset. The castle's walls are made of shimmering crystal, and the clouds around it glow with a warm, pastel pink and purple light. A gentle breeze blows, making the floating islands and delicate vines sway, adding to the dreamlike quality of the scene.",
#     "A whimsical garden with oversized flowers and glowing fireflies dancing in the air. The flowers are vibrant in colors of deep violet, pink, and turquoise, with petals that shimmer like diamonds. The background is a soft, pastel gradient sky with faint, glowing stars that add to the magical and serene atmosphere of the garden.",
#     "A surreal, floating island in the sky, with trees that have crystal leaves that sparkle in the soft light of a distant sun. The island is surrounded by swirling clouds and glowing butterflies. The sky is a gradient of pastel colors—lavender, mint, and soft gold—creating an otherworldly, dreamlike atmosphere. A glowing path winds through the island, leading to an unknown destination.",
#     "A serene lake with soft, glowing ripples under a sky filled with soft, billowy clouds. The air is misty, and a gentle breeze causes the water to shimmer with iridescent colors. Small, glowing jellyfish float just beneath the surface, casting soft light on the surrounding lily pads. The scene is peaceful, with a sense of calm and gentle beauty that feels like a dream.",
#     "A floating forest, where trees with glowing leaves gently sway in the breeze. The trees are connected by glowing bridges, and the atmosphere is filled with mist and soft light. The background features a pink and purple sky, with floating stars and soft, ethereal clouds. The scene feels like a magical dreamscape, suspended between reality and fantasy.",
#     "A vast meadow of delicate flowers that glow under the soft light of a full moon. The flowers are in soft shades of pink, blue, and white, creating a calming, peaceful effect. In the distance, a silhouette of a gentle unicorn stands in front of a glowing waterfall, casting a mystical light across the meadow. The entire scene feels like an enchanting dream come to life.",
#     "A celestial garden in space, where glowing plants and flowers bloom against the backdrop of distant stars and nebulae. The flowers glow with soft, pulsating light, and the plants sway gently in the cosmic breeze. The sky above is filled with swirling colors of pink, blue, and violet, creating a feeling of being in a dreamlike, otherworldly realm.",
#     "A mystical cave with walls made of glowing crystals, casting shimmering light across the scene. Soft, golden light filters through cracks in the ceiling, illuminating a gentle waterfall that falls into a glowing pool. The atmosphere is magical and peaceful, with delicate flowers growing at the base of the crystals, adding to the ethereal beauty of the cave.",
#     "A dreamlike sky filled with floating lanterns drifting upwards, casting a soft glow across a tranquil ocean. The water reflects the lanterns in a delicate dance of light, creating ripples of color that seem to shimmer in the moonlight. The atmosphere is peaceful and calming, as if time has slowed down, creating an unforgettable dreamlike experience."
# ]
    prompts = [
    "A colorful kite soaring in the clear blue sky, with its long, flowing tail fluttering in the breeze. Below, a grassy field stretches out, dotted with small wildflowers in shades of purple and yellow. The scene feels light, joyful, and carefree.",
    "A wooden rowboat gently floating on a calm lake, the water reflecting the soft pink and orange hues of the sunset. In the distance, tall green trees line the shore, and a few fluffy clouds float lazily across the sky.",
    "A cozy wooden cabin nestled in a snowy forest, smoke rising from the chimney. The warm light from the cabin’s windows contrasts with the cool blue and white of the snow, creating a peaceful, wintery scene.",
    "A simple bicycle leaning against a rustic fence, with a small basket of fresh flowers resting in the front. The background features a country road lined with tall trees, and the sun sets gently behind the scene, casting warm golden light.",
    "A stack of colorful books with their pages slightly bent, resting on a wooden desk. Beside the books, a small potted plant adds a touch of green. The background features soft, neutral tones and a window with sheer curtains, letting in soft natural light.",
    "A ceramic teapot with a floral design, placed on a delicate lace tablecloth. Steam rises gently from the spout, and the background features a softly blurred garden with colorful flowers in bloom, adding warmth and liveliness to the scene.",
    "A basket filled with ripe strawberries, resting on a wooden table. The bright red berries contrast beautifully with the green leaves and soft, natural light filtering through the background, creating a fresh, summery feel.",
    "A fluffy white cloud drifting across a pastel-colored sky at sunset, with shades of soft pink, lavender, and pale yellow blending together. Below, a calm beach with soft sand stretches out, inviting a sense of peace and serenity.",
    "A vintage suitcase sitting beside an open map, with a small compass resting on top. The background features a soft, cozy room with dimmed lights and a fireplace glowing in the distance, evoking a sense of adventure and nostalgia.",
    "A stack of freshly baked croissants, golden and flaky, placed on a simple white plate. Beside them, a steaming cup of coffee completes the breakfast scene. The background is softly blurred with light, warm beige tones, enhancing the cozy atmosphere.",
    # "A charming wooden birdhouse hanging from a tree branch, with small colorful birds perched around it. The background features a garden filled with blooming flowers and soft green grass, creating a lively and natural environment.",
    # "A single flamingo standing gracefully in shallow water, surrounded by tall grasses. The background features a soft sunrise with pastel hues of pink and blue, creating a tranquil, dreamlike atmosphere.",
    # "A small, rustic lantern with a flickering flame, resting on a wooden table beside a soft wool blanket. The background is softly blurred with the warm glow of a fireplace casting light over the scene, adding comfort and serenity.",
    # "A sleek, modern scooter parked on a quiet street lined with tall trees. The background features a vibrant cityscape with soft evening light filtering through the trees, creating a relaxed yet urban vibe.",
    # "A traditional Japanese umbrella resting against a stone lantern in a peaceful garden. The soft fabric of the umbrella contrasts with the stone and surrounding greenery, while the background features a tranquil pond and a few lotus flowers in bloom.",
    # "A small boat docked at a wooden pier, with the calm ocean reflecting the soft pastel colors of the early morning sky. The scene feels peaceful and serene, with a few seagulls flying in the distance, adding life to the quiet scene.",
    # "A basket of fresh bread loaves resting on a wooden countertop, with flour scattered around. The background features a soft kitchen scene with a few kitchen utensils and jars on a wooden shelf, evoking a warm, homely atmosphere.",
    # "A colorful parrot perched on a branch, with its vibrant feathers in shades of green, blue, and red. The background is a lush, tropical jungle with soft sunlight filtering through the leaves, creating a lively and exotic environment.",
    # "A pair of rain boots sitting on a stone path, with small puddles reflecting the sky above. In the background, a soft, misty forest is visible, with tall trees and fern-like plants adding texture and depth to the scene.",
    # "A hot air balloon floating in the sky, with its colorful fabric glowing in the early morning light. Below, a wide open field with soft, rolling hills creates a peaceful and expansive landscape.",
    # "A modern, minimalist living room bathed in warm natural sunlight. The room features a light wood coffee table in the center, with the surface left empty for a clean and tidy look. Next to the table is a deep green sofa with a simple, elegant design and rounded contours. Large windows in the background allow sunlight to flood in, creating a bright, welcoming atmosphere. Through the windows, a lush garden filled with greenery can be seen, slightly blurred to add a soft, natural effect. A tall, leafy green plant sits to the side of the sofa, enhancing the room’s fresh and relaxed vibe. The overall decor is simple yet aesthetically pleasing, emphasizing warmth, light, and a sense of natural beauty.",
    # "A small wooden bridge arching over a quiet stream, with smooth, rounded stones lining the bank. The background features a soft forest, with rays of sunlight filtering through the trees, adding a peaceful, natural feel to the scene."
]










    with tqdm(total=len(prompts), desc="Generating Images") as pbar:
        for idx in range(len(prompts)):
            prompt = prompts[idx:idx + 1]
            image_tokens = torch.ones((len(prompt), config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting((prompt, image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompt), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
                # print('1')
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
                # print('2')

            # mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "linear"))

            with torch.no_grad():
                seed=0
                generator = torch.Generator(device=device).manual_seed(seed)
                # generator=None
                temperature=config.training.get("generation_temperature", 1.0)
                # temperature=0
                noise_schedule=mask_schedule
                noise_type=config.training.get("noise_type", "mask")
                
                seq_len = config.model.showo.num_vq_tokens

                input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
                input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                            mask_token_id,
                                                            input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - 10)
                # import ipdb
                # ipdb.set_trace()
                if uncond_input_ids is not None:
                    uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]
                else:
                    uncond_prefix = None


                attention_mask_student=create_attention_mask_predict_next(input_ids,
                                                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                        rm_pad_in_image=True)
                uncond_input_ids_student = None
                uncond_prefix_student=None

                for step in range(config.training.generation_timesteps):
                    ratio = 1.0 * (step + 1) / config.training.generation_timesteps
                    input_ids, input_ids_minus_lm_vocab_size, temperature,sampled_ids = denoise(model,input_ids, input_ids_minus_lm_vocab_size, 
                                uncond_input_ids, uncond_prefix,attention_mask, config, 
                                generator, ratio, mask_token_id, noise_schedule,seq_len,temperature)
                    
                    
            gen_token_ids = sampled_ids
            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]
            # print(image_[0])
            pil_images[0].save(image_dir+ f"test_lmcm_x_photo_{idx}.png")
            # image_tensor = transforms.ToTensor()(image_[0])  
            # torch.save(image_tensor, image_dir+f'image_tensor_{idx}.pt') 
            # 更新进度条
            pbar.update(1)

    del model


    # def CS(texts, images):
    #     model, preprocess = clip.load("ViT-B/32", device=device)
    #     scores = []
        

    #     for text, image in tqdm(zip(texts, images), total=len(texts),desc="CS Processing images and prompts"):  # Wrap with tqdm
    #         img = preprocess(Image.open(image)).unsqueeze(0).to(device)
    #         text = clip.tokenize(text,truncate=True).to(device)

    #         logits_per_image, logits_per_text = model(img, text)
    #         score = logits_per_image.squeeze().item() 
    #         scores.append(score)
    #     del model
    #     del preprocess
    #     torch.cuda.empty_cache()
    #     return scores


    # # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # # 仅评测 style 为 photo 的部分
    # from datasets import load_dataset

    # mcm_dirs=[]
    # with tqdm(total=len(prompts), desc="Generating Images") as pbar:
    #     for idx, prompt in enumerate(prompts):
    #         image_p = image_dir+ f"test_lmcm_x_photo_{idx}.png"
    #         mcm_dirs.append(image_p)

    # # print(prompts)
    # # print(mcm_dirs)
    # result_mcm = hpsv2.score_my(mcm_dirs, prompts,device=device, hps_version="v2.1")
    # # print(result_mcm)
    # print('hps_mcm',sum(result_mcm)/len(result_mcm))

    # model = RM.load("ImageReward-v1.0",device=device)
    # score_mcm=[]
    # with torch.no_grad():
    #     for prompt, image_path in tqdm(zip(prompts, mcm_dirs),total=len(prompts),desc="IR Processing images and prompts"):
            
    #         score = model.score(prompt, image_path)
    #         score_mcm.append(score)

    # # print(score_mcm)
    # print('ir',sum(score_mcm)/len(score_mcm))

    # scores_CS=CS(prompts,mcm_dirs)
    # # print(scores_CS)
    # print('cs',sum(scores_CS)/len(scores_CS))


    # results = {
    #         "checkpoint_path": ck_path,
    #         "data_len": eval_len,
    #         "guidance_scale": guidance_scale,
    #         "inference_step": inference_step,
    #         "top_k": top_k,
    #         "hps_mcm": sum(result_mcm) / len(result_mcm),
    #         "ir": sum(score_mcm) / len(score_mcm),
    #         "cs": sum(scores_CS) / len(scores_CS),
    #         "description": discription,
    #     }

    # # Create the output directory if it doesn't exist
    # output_dir = os.path.join(os.path.dirname(ck_path), "evaluation_results") 
    # os.makedirs(output_dir, exist_ok=True)

    # # Write results to a JSONL file
    # output_file = os.path.join(output_dir, "results.jsonl")
    # with open(output_file, "a") as f:
    #     json.dump(results, f)
    #     f.write("\n")  # Add newline for JSONL format

    # # Clean up to free GPU memory after each checkpoint evaluation
    # torch.cuda.empty_cache()
