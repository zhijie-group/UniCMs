import matplotlib.pyplot as plt
import os
import sys
# os.chdir("/home/wx/Show-o")
sys.path.append("/home/chenkai/data/Show-o")
sys.path[0],sys.path[-1]=sys.path[-1],sys.path[0]
from accelerate.utils import DistributedType, set_seed
from lightning.pytorch.utilities import CombinedLoader
import os
import argparse
import logging
import math
import os
import sys
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftModel
import PIL
# os.chdir('/data/xck/Show-o')
# sys.path.append('/data/xck/Show-o')
import shutil
from pathlib import Path
from datasets import load_dataset
from accelerate import load_checkpoint_and_dispatch,infer_auto_device_map
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import AutoConfig
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
MAX_SEQ_LENGTH = 77
if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
logger = get_logger(__name__)
from tqdm import tqdm
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from models.modeling_utils import ConfigMixin, ModelMixin, register_to_config
from models.phi import PhiForCausalLM
from transformers import AutoTokenizer
import torch.nn.functional as F
from omegaconf import OmegaConf
from datasets import  load_dataset
import torchvision.transforms.functional as TF
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms

    
class CustomDataset(Dataset):#继承data.Dataset
    def __init__(self,t2i_path):
        with open(t2i_path,"r",encoding="utf-8") as f:
            self.data=json.load(f)
        self.resolution=256
        def transform(example, resolution=256, normalize=True):
            image=example["image"]
            image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
            image = transforms.CenterCrop((resolution, resolution))(image)
            image = transforms.ToTensor()(image)
            if normalize:
                image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
            example["image"]=image
            return example
        self.transform=transform
    # pass
    def __getitem__(self, index):
        # example={"text":self.data[index]["text"],"image":Image.open(self.data[index]["image"])}
        example={"text":self.data[index]["text"]}
        # print(example["text"])
        # self.transform(example)
        return example
    def __len__(self):
        return len(self.data)
class TextDataset(Dataset):
    def __init__(self,path="/data/wx/dataset/tiiuae/falcon-refinedweb/data",max_length=8000):
        self.ds = load_dataset("parquet",data_dir=path)["train"]
        self.max_length=max_length
        def transform(example):
            # resize image
            text = example["text"]
            text=text.replace("\n",'')
            if len(text) > max_length:
                start_index = random.randint(0, len(text) - self.max_length - 1)
                text = text[start_index:start_index + self.max_length]
            example["text"] = text
            return example
        self.transform=transform
    # pass
    def __getitem__(self, index):
        example={"text":self.ds[index]["content"]}
        self.transform(example)
        return example
    def __len__(self):
        return len(self.ds)
class MMUDataset(Dataset):
    def __init__(self,path="/data/wx/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",image_path="/home/chenkai/data/g3_dataset/llava_train_image/"):
        with open(path,"r",encoding="utf-8") as f:
            self.data=json.load(f)
        self.resolution=256
        self.image_path=image_path
        def transform(example):
            # resize image

            image = example["image"]
            image = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
            image = transforms.CenterCrop((self.resolution, self.resolution))(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

            example["image"] = image
            return example
        self.transform=transform
    # pass
    def __getitem__(self, index):
        ori_idx = index#
        while True:
                
                d = self.data[ori_idx]
                try:
                    idx=random.randint(0,len(d["conversations"])//2-1)
                    prompt = d["conversations"][2*idx]["value"]
                    answer = d["conversations"][2*idx+1]["value"]
                    image_path = self.image_path + d["image"]

                    # Load the image
                    image_ori = Image.open(image_path).convert("RGB")
                    
                    # Prepare the example
                    example = {"text": prompt, "image": image_ori,"answer":answer}
                    example = self.transform(example)
                    return example

                except (KeyError, FileNotFoundError, OSError, PIL.UnidentifiedImageError) as e:
                    # print(f"Error loading data item {d}: {e}")
                    ori_idx=random.randint(0, len(self.data) - 1)
                    continue  # Skip to the next data item if there's an erro
    def __len__(self):
        return len(self.data)
def create_dataloader(dataset:Dataset,per_gpu_batch_size,num_workers,num_train_examples,global_batch_size):
# 创建 RandomSampler 以实现随机打乱
    sampler = torch.utils.data.RandomSampler(dataset)

    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        drop_last=True,
        sampler=sampler  # 使用 RandomSampler
    )

    num_batches = math.ceil(num_train_examples / global_batch_size)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_train_examples

    return dataloader



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_student_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM student model or model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    # )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--unet_time_cond_proj_dim", type=int, default=512, help="The time embedding projection dimension for the student U-Net.")
    parser.add_argument("--train_type", type=str, default="distillation", help="The type of training to perform.")

    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA adaptation.")



    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--num_train_inferences", type=int, default=8, help="Number of inferences to run during training.")

    parser.add_argument("--image_dir", type=str, default="/home/chenkai/data/image/ckimage", help="The directory to save the generated images.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args
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
    try:
        attention_mask = attention_mask.to(dtype=model.module.dtype)
    except AttributeError:
        attention_mask = attention_mask.to(dtype=model.dtype)

    if input_embeddings is None:
        try:
            logits = model.module.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        except AttributeError:
            logits = model.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
    else:
        try:
            logits = model.module.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']
        except AttributeError:
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

def teacher_denoise(model,input_ids, input_ids_minus_lm_vocab_size,uncond_input_ids, uncond_prefix,
            attention_mask, config, generator, ratio, mask_token_id, noise_schedule,seq_len,temperature,
            return_logits=False,return_sampled_ids=False,return_sampled=False,return_masking=False,fix_reduce=False):
    with torch.no_grad():
        with torch.autocast("cuda"):
            if uncond_input_ids is not None and config.training.guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = forward(model,model_input.to(model.device), attention_mask=attention_mask.to(model.device)).chunk(2)
                cond_logits.to(accelerator.device)
                uncond_logits.to(accelerator.device)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has different cfg setting
                config.training.guidance_scale=10
                logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
                logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
            else:
                logits = forward(model,input_ids, attention_mask=attention_mask)
                logits.to(accelerator.device)
                input_ids.to(accelerator.device)
                attention_mask.to(accelerator.device)
                logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]

            # print(logits)

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            if return_sampled==True:
                return sampled
            # print(generator.get_state())
            # sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            sampled_ids = torch.argmax(sampled, dim=-1).view(*logits.shape[:-1])
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map.to(accelerator.device), sampled_ids.to(accelerator.device), input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            
            # print(ratio)
            # print(noise_schedule)
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs.to(accelerator.device), -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)
            # print(selected_probs)
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(accelerator.device)

            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            unknown_map.to(accelerator.device)
            mask_len = torch.max(
                torch.tensor([0], device=accelerator.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            # print(mask_len,temperature)
            # print(mask_len.shape)
            # print(selected_probs.shape)
            # print(temperature.shape)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(seq_len + 1):-1] = torch.where(masking, mask_token_id,
                                                            sampled_ids + config.model.showo.llm_vocab_size + 10)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            if return_logits:
                return input_ids, input_ids_minus_lm_vocab_size, temperature,unknown_map,masking, sampled 
            elif return_sampled_ids:
                return input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids,unknown_map,masking, sampled 
            else:
                return input_ids, input_ids_minus_lm_vocab_size, temperature
    

def denoise(model,input_ids, input_ids_minus_lm_vocab_size,uncond_input_ids, uncond_prefix,
            attention_mask, config,mask_token_id,seq_len,generator,noise_schedule,ratio,temperature,
            return_logits=False,return_sampled_ids=False,return_sampled=False,remove_mask=False):
    if uncond_input_ids is not None and config.training.guidance_scale > 0:
        uncond_input_ids = torch.cat(
            [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
        model_input = torch.cat([input_ids, uncond_input_ids])
        cond_logits, uncond_logits = forward(model,model_input, attention_mask=attention_mask).chunk(2)
        # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        # it seems that muse has different cfg setting
        student_guide=0
        logits = (1 + student_guide) * cond_logits - student_guide * uncond_logits
        # print("logits1",logits.shape)
        logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
        cond_logits = cond_logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
        # print("logits2",logits.shape)
    else:
        logits = forward(model,input_ids, attention_mask=attention_mask)
        logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]

    # print(logits)

    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))

    # cond_probs = cond_logits.softmax(dim=-1)
    cond_logits = cond_logits.reshape(-1, cond_logits.size(-1))
    # print(generator.get_state())
    
    
    # print(generator.get_state())
    # sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
    sampled_ids = torch.argmax(sampled, dim=-1).view(*logits.shape[:-1])
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
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
    # Masks tokens with lower confidence.
    input_ids[:, -(seq_len + 1):-1] = torch.where(masking, mask_token_id,
                                                    sampled_ids + config.model.showo.llm_vocab_size + 10)
    input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

    if return_sampled:
        return sampled,masking

    if return_logits:
        return unknown_map,masking, sampled, cond_logits 
    elif return_sampled_ids:
        print("mask总数",masking.sum())
        mask_probs=sampled[:,-1]
        print("mask最大概率",mask_probs.max(),mask_probs.mean(),mask_probs.min())
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
        selected_probs = selected_probs.squeeze(-1)
        print("全部最大概率",selected_probs.max())
        return input_ids, input_ids_minus_lm_vocab_size,temperature, sampled_ids
    else:
        return input_ids, input_ids_minus_lm_vocab_size


def sample_and_save_image(uni_prompting,step, sample_prompt, vq_model, model, mask_schedule, mask_token_id,sample_steps=4):
    gen_token_ids=sampling_from_multistep_consistency(uni_prompting,sample_prompt, vq_model, model, 
                            mask_schedule,mask_token_id, sample_steps=sample_steps)

    from PIL import Image
    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids.to(vq_model.device))
    images.to(model.device)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    output_file=args.image_dir+ f'/sample_step_{step}.png'
    pil_images[0].save(output_file)

def sampling_from_multistep_consistency(uni_prompting,sample_prompt, vq_model, model, mask_schedule, mask_token_id,sample_steps):
    # 1. Create a random noise tensor z_T sampled from a normal distribution


    seed=42
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)  # 使用种子来保证结果一致
    prompts=[sample_prompt]
    # mask_token_id = model.module.config.showo.mask_token_id
    image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                            dtype=torch.long, device=accelerator.device) * mask_token_id

    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
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

    temperature=config.training.get("generation_temperature", 1.0)
    

    input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
    input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                mask_token_id,
                                input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - 10)
    

    if uncond_input_ids is not None:
        uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]
    
    
    timesteps = torch.linspace(sample_steps / sample_steps, 1 / sample_steps, sample_steps, device=accelerator.device)
    
    for t_ in tqdm(timesteps, desc="Sampling steps"):
        s_ = t_ - 1 / sample_steps
        ratio_s=1-s_
        with torch.no_grad():
            # Compute f(z_t, t)
            
            input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids= denoise(model,
                    input_ids, input_ids_minus_lm_vocab_size, 
                    uncond_input_ids, uncond_prefix,attention_mask, config, mask_token_id,seq_len,
                    generator,mask_schedule, ratio_s,temperature,return_logits=False,return_sampled_ids=True)
            
    return sampled_ids


def teacher_sample_and_save_image(uni_prompting,step, sample_prompt, vq_model, model, mask_schedule, mask_token_id,sample_steps=18):
    gen_token_ids=teacher_sampling_from_multistep_consistency(uni_prompting,sample_prompt, vq_model, model, 
                            mask_schedule,mask_token_id, sample_steps=sample_steps)

    from PIL import Image
    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids.to(vq_model.device))
    images.to(model.device)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    output_file=args.image_dir+ f'/teacher_sample_step_{step}.png'
    pil_images[0].save(output_file)

def teacher_sampling_from_multistep_consistency(uni_prompting,sample_prompt, vq_model, model, mask_schedule, mask_token_id,sample_steps):
    # 1. Create a random noise tensor z_T sampled from a normal distribution


    seed=42
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)  # 使用种子来保证结果一致
    prompts=[sample_prompt]
    # mask_token_id = model.module.config.showo.mask_token_id
    image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                            dtype=torch.long, device=accelerator.device) * mask_token_id

    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
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

    temperature=config.training.get("generation_temperature", 1.0)
    

    input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
    input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                mask_token_id,
                                input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - 10)
    

    if uncond_input_ids is not None:
        uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]
    
    
    # 2. Define the timesteps for the loop
    timesteps = torch.linspace(sample_steps / sample_steps, 1 / sample_steps, sample_steps, device=accelerator.device)
    
    for t_ in tqdm(timesteps, desc="Sampling steps"):
        s_ = t_ - 1 / sample_steps
        ratio_s=1-s_
        with torch.no_grad():
            # Compute f(z_t, t)
            
            input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids= teacher_denoise(model,
                    input_ids, input_ids_minus_lm_vocab_size, 
                    uncond_input_ids, uncond_prefix,attention_mask, config, 
                    generator, ratio_s, mask_token_id,mask_schedule,seq_len,temperature,return_sampled_ids=True)
      
    return sampled_ids



def generate_intermediate_t_vectors(steps,step,t,bsz,device,seq_len):
    rate=1
    # Create a tensor to hold all intermediate t vectors for each batch element
    intermediate_ts = torch.zeros(rate*(steps - step[0].item())+1, bsz, device=device)

    # Calculate intermediate values for t for each batch element
    for i in range(bsz):
        end_t = t[i].item()
        start_t = seq_len
        num_intervals = rate*(steps - step[i].item())+1
        if num_intervals > 0:
            # Create evenly spaced values between start_t and end_t
            intermediate_ts[:num_intervals, i] = torch.linspace(start_t, end_t, num_intervals, device=device)

    return intermediate_ts



    

args = parse_args()
# teacher_model = PixArtmodel2DModel.from_pretrained(args.pretrained_teacher_model, subfolder="model")



# # 2. 加载 text encoder
# text_encoder = T5EncoderModel.from_pretrained(args.pretrained_teacher_model, subfolder="text_encoder",device_map="balanced_low_0")
# print("text_encoder.device",text_encoder.device)
# # 3. 加载 VAE
# vae = AutoencoderKL.from_pretrained(args.pretrained_teacher_model, subfolder="vae",device_map="balanced_low_0")

# # 4. 加载 model


logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

config_file_path = "/home/wx/Show-o/configs/showo_demo.yaml"
config = OmegaConf.load(config_file_path)
config.mode='t2i'

seq_len = config.model.showo.num_vq_tokens

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

tokenizer= AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
    special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

print("uni_prompting",len(uni_prompting.text_tokenizer))


import random
def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, timestep,is_train=True):
    batch_size, seq_len = image_tokens.shape

    
    # Sample a random timestep for each image
    timesteps = torch.tensor([timestep for i in range(batch_size)],dtype=torch.float)
    # Sample a random mask probability for each image using timestep and cosine schedule
    mask_prob = mask_schedule(timesteps)
    mask_prob = mask_prob.clip(0.0)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        num_token_masked=num_token_masked.to(batch_randperm.device)
        # print(batch_randperm.device,num_token_masked.device,type(num_token_masked))
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, mask, loss_weight, mask_prob
def prepare_inputs_and_labels(
        vq_model,
        pixel_values_or_image_ids ,
        texts,
        mask_id,
        mask_schedule,
        timestep,
        min_masking_rate: float = 0.0,
        is_train: bool = True,
        
):
    # print(vq_model.device,pixel_values_or_image_ids.device)
    pixel_values_or_image_ids=pixel_values_or_image_ids.to(torch.float16)
    image_tokens = vq_model.get_code(pixel_values_or_image_ids.to(vq_model.device))

    image_tokens_ori=image_tokens.clone()
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
    # print(len(uni_prompting.text_tokenizer))
    # create MLM mask and labels
    input_ids, mask, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        timestep=timestep,
        is_train=is_train,
    )
    # print("mask_prob",mask_prob)
    image_tokens=input_ids
    input_ids, _ = uni_prompting((texts, input_ids), 't2i_gen')

    return input_ids, image_tokens,image_tokens_ori

def create_attention_mask_for_mmu(sequence, eoi_id=128258, return_inverse_mask=True):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    eoi_image = torch.where(sequence == eoi_id)[1]
    causal_mask[:, :, :, :eoi_image[0] + 1] = 1

    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask
    else:
        return causal_mask
    

def calculate_lm_loss(model, input_ids,input_embeddings, attention_mask, labels, label_smoothing,max_seq_length):
    input_ids=input_ids.to(model.device)
    attention_mask=attention_mask.to(model.device)
    labels=labels.to(model.device)
    # print("input_ids",input_ids.device)
    # print("attention_mask",attention_mask.device)
    # print("labels",labels.device)
    # print("model.device",model.device)
    logits=model.module.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
    loss_lm = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, model.module.output_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100,
            )

    return logits, loss_lm





def global_image_process(teacher_model,model,ref_num, input_ids,input_ids_minus_lm_vocab_size,uncond_input_ids, uncond_prefix,attention_mask, config, 
                    generator,mask_token_id,mask_schedule,seq_len,temperature,step,sample_steps=16,num=8193):
    # 2. Define the timesteps for the loop
    timesteps = torch.linspace(sample_steps / sample_steps, 1 / sample_steps, sample_steps, device=accelerator.device)
    label_step=torch.linspace(sample_steps/ref_num, sample_steps, ref_num, device=accelerator.device)
    global_image=[]
    global_id=[]
    global_image.append(input_ids_minus_lm_vocab_size.clone())
    global_id.append(input_ids.clone())
    mask_ratios=[]
    labels=[]
    r_len=sample_steps//ref_num
    label_id=step//r_len
    total_sampled=0
    teacher_total_sampled=0
    for idx,t_ in enumerate(timesteps):
        s_ = t_ - 1 / sample_steps
        ratio_s=1-s_
        mask_ratios.append(mask_schedule(torch.tensor(ratio_s)))
        with torch.no_grad():
            # Compute f(z_t, t)
            input_ids_student=input_ids.clone()
            input_ids_minus_lm_vocab_size_student=input_ids_minus_lm_vocab_size.clone()

            input_ids, input_ids_minus_lm_vocab_size, temperature,sampled_ids,unknown_map,teacher_masking, teacher_sampled = teacher_denoise(teacher_model,
                    input_ids, input_ids_minus_lm_vocab_size, 
                    uncond_input_ids, uncond_prefix,attention_mask, config, 
                    generator, ratio_s, mask_token_id,mask_schedule,seq_len,0,return_sampled_ids=True)
            
            teacher_sampled=teacher_sampled.reshape(-1,teacher_sampled.shape[-1])
            unknown_map=unknown_map.reshape(-1)
            known_map=~unknown_map
            teacher_sampled=teacher_sampled.to(unknown_map.device)
            teacher_total_sampled=teacher_sampled*unknown_map.unsqueeze(-1)+teacher_total_sampled*known_map.unsqueeze(-1)
            

            sampled,masking=denoise(model,input_ids_student, input_ids_minus_lm_vocab_size_student,uncond_input_ids, uncond_prefix,
                attention_mask, config,mask_token_id,seq_len,generator,mask_schedule,ratio_s,0,return_sampled=True)

            global_image.append(input_ids_minus_lm_vocab_size.clone())
            global_id.append(input_ids.clone())

            sampled=sampled.reshape(-1,sampled.shape[-1])
            unknown_map=unknown_map.reshape(-1)
            known_map=~unknown_map
            sampled=sampled.to(unknown_map.device)
            total_sampled=sampled*unknown_map.unsqueeze(-1)+total_sampled*known_map.unsqueeze(-1)

            if idx in label_step:
                labels.append((total_sampled,teacher_total_sampled,unknown_map))
                # if len(labels)==label_id+1:
                #     break
                
    labels.append((total_sampled,teacher_total_sampled,unknown_map))

        
    
    label=labels[label_id]
    teacher_label=sampled_ids.reshape(-1)

    return label,teacher_label,global_image,global_id

def detect_repetitive_patterns(tokenizer, prompt_ids, repeat_ngram_size):

    if len(prompt_ids.shape)==1:
        prompt_ids = prompt_ids
    elif len(prompt_ids.shape)==2:
        prompt_ids = prompt_ids[0]
    elif len(prompt_ids.shape)==3:
        prompt_ids = prompt_ids[0][0]
    else:
        print(f'Unexpected shape {prompt_ids.shape}! Please check prompt ids')
        assert False

    count = 1
    for i in range(1, len(prompt_ids)):
        if prompt_ids[i] == tokenizer.eos_token_id:
            break
        if prompt_ids[i] == prompt_ids[i - 1]:
            count += 1
            if count == repeat_ngram_size:
                return True
        else:
            count = 1

    return False

@torch.inference_mode()
def get_jacobian_trajectory(
    teacher_model,
    tokenizer,
    input_ids,
    teacher_mask_type,
    max_new_tokens
    ):

    bsz = input_ids.shape[0]
    prompt_len=[input_ids[i].shape[0] for i in range(bsz)]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # print("bsz:",bsz,"total_len:",total_len)

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device=teacher_model.device)

    for i in range(bsz):
        max_index = len(uni_prompting.text_tokenizer) - 1
        filtered_choices = [x for x in input_ids[i] if 0 <= x <= max_index]
        tokens[i, :] = torch.tensor(random.choices(filtered_choices, k=total_len)).to(dtype=torch.long, device="cuda")
        # tokens[i, :] = torch.tensor(random.choices(input_ids[i], k=total_len)).to(dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")


    
    trajectory = []
    logits_trajectory = []
    next_generation = tokens
    generate_attention_mask = create_attention_mask_for_mmu(next_generation.to(teacher_model.device),
                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
    generate_attention_mask = generate_attention_mask.to(teacher_mask_type)
    trajectory.append(tokens)
    itr=0
    while True:
        
        current_generation = next_generation
        # current_text=uni_prompting.text_tokenizer.batch_decode(current_generation, skip_special_tokens=True)
        # print("current_text:",current_text)
        # print("current_generation:",current_generation.dtype)
        # print("generate_attention_mask:",generate_attention_mask.dtype)
        logits = teacher_model(current_generation, attention_mask=generate_attention_mask)

        logits_trajectory.append(logits)

        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)


        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)

        next_text=tokenizer.decode(next_generation.tolist()[0])
        # print("itr",itr,"next_text:",next_text)

        trajectory.append(next_generation)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            
            # eos_reached = len(torch.where(trajectory[-1][0] == tokenizer.eos_token_id)[0])>0
            # for i in range(len(trajectory[-1][0])): 
            #     if trajectory[-1][0][i]==tokenizer.eos_token_id:
            #         print("eos_index:",i)
            # print("idx:",torch.where(trajectory[-1][0] == tokenizer.eos_token_id))
            eos_idxs=torch.where(trajectory[-1][0] == tokenizer.eos_token_id)
            eos_reached = len(eos_idxs[0])>1
            # print("shape:",trajectory[-1].shape)
            # print("eos_reached:",eos_reached)
            # print("eos",tokenizer.decode([tokenizer.eos_token_id]))
            return trajectory[:-1], logits_trajectory[-1], eos_reached # converged generation is saved twice so we delete the last element of trajectory list
        itr+=1

def clean_prompt(prompt):
    # 移除换行符
    cleaned_prompt = prompt.replace('\n', ' ')
    # 移除 <image>
    cleaned_prompt = cleaned_prompt.replace('<image>', '')
    return cleaned_prompt

from typing import Dict
def preprocess_distill_data(
    prompt_ids,
    answer_trajectory_ids,
    teacher_output_ids,
    tokenizer,
    model: str,
    labels_ids=None,
) -> Dict:
    
    jacobian_trajectory_ids = []
    # only take batch size 1 for now
    # TODO: support bsz > 1 from the generation script. for now, only prompt ids is in (bsz, seq_len)
    # jacobian_prompt_ids = torch.tensor(prompt_ids[0], dtype=torch.int64)
    # teacher_output_ids = torch.tensor(teacher_output_ids[0], dtype=torch.int64)
    jacobian_prompt_ids=prompt_ids[0]
    teacher_output_ids=teacher_output_ids[0]
    for answer_ids in answer_trajectory_ids:
    #     answer_ids = torch.tensor(answer_ids, dtype=torch.int64)
        #print(answer_ids)
        #print(jacobian_prompt_ids)
        if len(jacobian_prompt_ids.shape) == len(answer_ids.shape):
            trajectory_ids = torch.cat((jacobian_prompt_ids, answer_ids), dim=-1)
        elif len(jacobian_prompt_ids.shape) > len(answer_ids.shape):
            #print(f'prompt: {jacobian_prompt_ids.shape}')
            #print(f'answer: {answer_ids.shape}')
            trajectory_ids = torch.cat((jacobian_prompt_ids[0], answer_ids), dim=-1)
        # print(trajectory_ids.shape) # torch.Size([228])
        jacobian_trajectory_ids.append(trajectory_ids.unsqueeze(0))
   
    if labels_ids:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            labels_ids=labels_ids,
            teacher_output_ids=teacher_output_ids,
        )
    else:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            teacher_output_ids=teacher_output_ids,
        )

def soft_cross_entropy(predicts, targets, padding_mask):
    # TODO: support batch_size >1 here.
    if (~padding_mask).sum() == 0:
        return 0*predicts[0][0][0]
    predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    entropy = -targets_prob * predict_log_prob
    expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
    entropy.masked_fill_(expand_mask, 0)
    mean_entropy = entropy.sum() / (~padding_mask).sum()
    return mean_entropy

def calculate_cllm_loss(last_jacobian, last_attention_mask, pick_jacobian, pick_attention_mask, output_mask,labels,labels_attention_mask,ignore_labels,model, tokenizer):
    last_jacobian = last_jacobian.to(model.device)
    last_attention_mask = last_attention_mask.to(model.device)
    pick_jacobian = pick_jacobian.to(model.device)
    pick_attention_mask = pick_attention_mask.to(model.device)
    output_mask = output_mask.to(model.device)
    labels = labels.to(model.device)
    labels_attention_mask = labels_attention_mask.to(model.device)
    ignore_labels=ignore_labels.to(model.device)

    last_logits=model.module.showo(input_ids=last_jacobian, attention_mask=last_attention_mask)['logits']
    pick_logits=model.module.showo(input_ids=pick_jacobian, attention_mask=pick_attention_mask)['logits']
    labels_logits=model.module.showo(input_ids=labels, attention_mask=labels_attention_mask)['logits']

    loss_global = soft_cross_entropy(
                    pick_logits[..., :-1, :].float(), # logits generated by the last token is dropped
                    last_logits[..., :-1, :].clone().detach().float(),
                    output_mask.to(pick_logits.device)
        )
    
    label_logits=labels_logits.view(-1, labels_logits.size(-1))
    ignore_labels=ignore_labels.view(-1)
    loss_ar= F.cross_entropy(label_logits[:-1], ignore_labels[1:], ignore_index=-100)
    loss_ar*=10
    loss_global+=loss_ar

    return loss_global

def main(args):
    device=accelerator.device
    total_batch_size_per_gpu = args.train_batch_size
    if accelerator.distributed_type == DistributedType.DEEPSPEED:

        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.image_dir is not None:
            os.makedirs(args.image_dir, exist_ok=True)
            
        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id


    weight_dtype = torch.float16
    
    # print("vq_model")
    # os.system("gpustat")
    
    
    # teacher_model=distributed_model("teacher_transformer",weight_dtype,config,args)
    teacher_model = Showo.from_pretrained(args.pretrained_teacher_model)
    # print(os.system("gpustat"))
    # teacher_model=load_checkpoint_and_dispatch(teacher_model,
    #     checkpoint="/home/xck/data/models/showo/show-o/pytorch_model.safetensors",device_map="balanced_low_0",no_split_module_classes=no_split_module)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    teacher_model.to(accelerator.device).to(weight_dtype)
    # print("teacher_model")
    # os.system("gpustat")

    # vq_model=distributed_model("vq_model",weight_dtype,config,args)
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model =vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.requires_grad_(False)
    vq_model.eval()
    vq_model.to(accelerator.device).to(weight_dtype)
    student_model = Showo.from_pretrained(args.pretrained_student_model)

    model = Showo.from_config(student_model.config)
    model.load_state_dict(student_model.state_dict())
    del student_model


    # lora_config = LoraConfig(
    #     r=256,
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "dense",
    #         "fc1",
    #         "fc2",
    #         "lm_head",
    #         # "scale_shift_table",      # not available due to the implementation in huggingface/peft, working on it.
    #     ],
    # )

    for param in model.parameters():
        param.requires_grad = True

    # 解冻需要使用 LoRA 训练的层
    # for name, module in model.named_modules():
    #     # print(name)
    #     if any([target_module in name for target_module in lora_config.target_modules]):
    #         for param in module.parameters():
    #             param.requires_grad = True

    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    # lora_layers = filter(lambda p: p.requires_grad, model.parameters())
    # print("model",model)
    # num_layers_to_train = 1
    # for layer in model.showo.model.layers[-num_layers_to_train:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True

    # for param in model.showo.model.final_layernorm.parameters():
    #     param.requires_grad = True
    # for param in model.showo.lm_head.parameters():
    #     param.requires_grad = True

    # trainable_params = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         trainable_params.append(name)
    #         print(f"可训练参数: {name}")
    # print(f"总共有 {len(trainable_params)} 个可训练参数")


    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    train_layers = filter(lambda p: p.requires_grad, model.parameters())

    torch.cuda.empty_cache()
    model.train()


    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(model).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(model).dtype}. {low_precision_error_string}"
        )
    
    # print(vae.device)
    # os.system("gpustat")
    # teacher_model=teacher_model.to(weight_dtype).to(device)
    # os.system("gpustat")
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    
    # vae=vae.to(weight_dtype).to(accelerator.device)
    # text_encoder.to(accelerator.device)
    # print(model.device,text_encoder.device,vae.device)
    # os.system("gpustat")
    for param in model.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)


    # ck_path="/home/xck/data/ckpt/showo_softlabel_linebias20_0004002_1e-5/checkpoint-5800"
    # input_dir=ck_path+"/model"
    # load_model = Showo.from_pretrained(input_dir)
    # model.register_to_config(**load_model.config)

    # model.load_state_dict(load_model.state_dict())
    # del load_model

    # os.system("gpustat")
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    save_path / "unwrapped_model",
                    save_function=accelerator.save,
                    state_dict=state_dict,
                    safe_serialization=False
                )
            # if accelerator.is_main_process:

            #     for i, model in enumerate(models):
            #         # 提取包含更新参数的基础模型
            #         # save_model = model.merge_and_unload()
            #         # print("model",model)
            #         model.save_pretrained(os.path.join(output_dir, "model"))

            #         weights.pop()


        def load_model_hook(models, input_dir):
            # base_dir = os.path.join(input_dir, "base_model")
            input_dir = os.path.join(input_dir, "model")
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                
                print("input_dir",input_dir)
                # load diffusers style into model
                
                load_model = Showo.from_pretrained("/data/xck/models/showo/show-o")
                peft_config = LoraConfig.from_pretrained(input_dir)
                load_model = PeftModel.from_pretrained(load_model,input_dir, config=peft_config)  
                model.register_to_config(**load_model.config)
 
                model.load_state_dict(load_model.state_dict())
                del load_model

        # accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(
        train_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = CustomDataset(
        t2i_path=args.train_shards_path_or_url,
    )
    text_dataset=TextDataset()
    mmu_dataset=MMUDataset()
    train_dataloader =create_dataloader(dataset,  num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,)
    text_train_dataloader =create_dataloader( text_dataset,  num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,)
    mmu_train_dataloader=create_dataloader(mmu_dataset,  num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,)
    iterables = {
    "t2i_flow": train_dataloader,
    "lm_flow": text_train_dataloader,
    "mmu_flow": mmu_train_dataloader,
}

    combined_dataloader = CombinedLoader(iterables, mode="min_size")
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )


    
    model ,optimizer, lr_scheduler ,combined_dataloader= accelerator.prepare(model ,optimizer, lr_scheduler,combined_dataloader)
    # os.system("gpustat")
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # os.system("gpustat")
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.learning_rate
                
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # print('steps in one epoch',expect_len)


    mask_token_id = model.module.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = args.train_batch_size
    config.training.guidance_scale = 1.75

    
    steps=args.num_train_inferences
    steps=16
    segment_num=1
    t2i_ref_num=1

    Tstep=seq_len/steps
    Tstep=math.ceil(Tstep)
    print('Tstep',Tstep)
    batch_size=args.train_batch_size

    print('first_epoch',first_epoch)
    torch.cuda.empty_cache()
    loss1_ema=0.0
    loss2_ema=0.0
    loss3_ema=0.0
    loss_mmu_ema=0.0
    ema_alpha=0.9
    step_losses = {}
    num_pred_dict = {}
    num_hope_dict = {}
    acc_dict = {}
    ignore_len=seq_len+4
    ignore_id=-100
    max_new_tokens=16
    max_new_seq_len=512
    use_gt=False
    mask_type=model.module.showo.model.embed_tokens.weight.dtype
    teacher_mask_type=teacher_model.showo.model.embed_tokens.weight.dtype

    
    # os.system("gpustat")
    for batch, batch_idx, dataloader_idx in combined_dataloader:
        batch_mmu=batch["mmu_flow"]
        batch_t2i=batch["t2i_flow"]
        batch_lm=batch["lm_flow"]
        with accelerator.accumulate(model):
            # weights = torch.arange(steps, 0, -1, device=device).float()
            # probabilities = weights / weights.sum()
            # step = torch.multinomial(probabilities, num_samples=1).long()
            # ori_idx=(global_step+50000)%len_data
            image = batch_mmu["image"]
            prompt=batch_mmu["text"][0]
            answer=batch_mmu["answer"][0]
            image=image.to(vq_model.dtype).to(accelerator.device)

            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            batch_size = 1

            if len(prompt) > 1024:
                print(f"Skipping data item: Prompt too long.")
                continue

            prompt = clean_prompt(prompt)
            input_text = ['USER: \n' + prompt + ' ASSISTANT:']
            label_text = ['USER: \n' + prompt + ' ASSISTANT: ' + answer]

            input_ids = uni_prompting.text_tokenizer(input_text)['input_ids']
            input_ids = torch.tensor(input_ids).to(device)
            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

            inputs=input_ids.to(model.device)

            iitr = 0
            eeos_reached=False
            while iitr * max_new_tokens < max_new_seq_len and eeos_reached==False:
                dic = {}
                dic['prompt_ids_len'] = len(inputs[0])
                # print("inputs:",inputs.shape)
                dic['prompt_ids'] = inputs

                with torch.no_grad():
                    jacobian_trajectory_ids, teacher_logits, eeos_reached = get_jacobian_trajectory(teacher_model, tokenizer, inputs, teacher_mask_type, max_new_tokens)
                
                dic["answer_trajectory_ids"] = []
                for _, id in enumerate(jacobian_trajectory_ids):
                    # only support batch size=1 now
                    dic["answer_trajectory_ids"].append(id[0][-max_new_tokens:])
                dic['teacher_output_ids'] = jacobian_trajectory_ids[-1]

                inputs = jacobian_trajectory_ids[-1]
                iitr+=1


                low_data=detect_repetitive_patterns(tokenizer, prompt_ids=inputs, repeat_ngram_size=10)
                if low_data:
                    print("low quality data is detected")
                    break
        
                train_d = preprocess_distill_data(dic["prompt_ids"],
                         dic["answer_trajectory_ids"],
                         dic["teacher_output_ids"],
                         tokenizer,
                         model,
                )

                jacobian_trajectory = train_d["jacobian_trajectory"]


                
                # print("len(jacobian_trajectory)",len(jacobian_trajectory))
                segment_length = round(len(jacobian_trajectory) / segment_num)
                segments = [jacobian_trajectory[i:i + segment_length] for i in range(0, len(jacobian_trajectory), segment_length)]
                i = random.choice(range(len(jacobian_trajectory))[:-1])

                # 找到该索引所在的段
                segment_index = i // segment_length

                # 找到段末尾对应的索引
                end_index_of_segment = (segment_index + 1) * segment_length - 1
                if end_index_of_segment >= len(jacobian_trajectory):
                    end_index_of_segment = len(jacobian_trajectory) - 1 


                last_jacobian = jacobian_trajectory[end_index_of_segment].clone().detach()
                last_attention_mask = create_attention_mask_for_mmu(last_jacobian.to(device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                last_attention_mask=last_attention_mask.to(mask_type)
                
                pick_jacobian = jacobian_trajectory[i].clone().detach()
                pick_attention_mask = create_attention_mask_for_mmu(pick_jacobian.to(device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                pick_attention_mask=pick_attention_mask.to(mask_type)
                output_mask = jacobian_trajectory[i][..., 1:] == tokenizer.pad_token_id
                # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
                for j in range(1):
                    end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[end_index_of_segment][j, 1:])[0]
                    if len(end_of_mask_position)==0:
                        output_mask[j, :] = True
                    else:
                        output_mask[j, :end_of_mask_position[0]] = True



                labels = train_d['teacher_output_ids']

                
                labels=labels.unsqueeze(0)
                labels = torch.tensor(labels).to(model.device)
                ignore_labels=labels.clone().detach()
                ignore_len=seq_len+4
                ignore_labels[0][:ignore_len] = ignore_id

                labels_attention_mask = create_attention_mask_for_mmu(labels.to(device),
                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                labels_attention_mask=labels_attention_mask.to(mask_type)
                loss_mmu=calculate_cllm_loss(last_jacobian, last_attention_mask, pick_jacobian, pick_attention_mask, output_mask, labels,labels_attention_mask,ignore_labels,model, tokenizer)



               
                # Get the batch corresponding to t2i_index
                # try:
                #     batch = next(data_iter)
                # except StopIteration:
                #     data_iter = iter(train_dataloader)
                #     batch = next(data_iter)

                prompts =batch_t2i['text']
                # images=batch_t2i["image"]

                step = torch.randint(0, steps, (1,), device=device).long()
                # Compute initial tstep and t
                tstep = step.float() / steps 
                # nrel = torch.randint(0, Tstep, (1,), device=device).long()


                timestep=0
                # input_ids_data,image_tokens_data,label=prepare_inputs_and_labels(vq_model,images,prompts,mask_token_id,mask_schedule,timestep)
                generator=None
                temperature=config.training.get("generation_temperature", 1.0)


                image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id

                input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
                # os.system("gpustat")
                if config.training.guidance_scale > 0:
                    uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
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

                input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
                input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                            mask_token_id,
                                            input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - 10)
                

                if uncond_input_ids is not None:
                    uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

                # intermediate_t_vectors = generate_intermediate_t_vectors(steps,0,0,1,device,seq_len)
                # intermediate_t_vectors = intermediate_t_vectors.to(torch.int64)


                label_mask,teacher_label,global_image,global_id=global_image_process(teacher_model,model,t2i_ref_num,input_ids, input_ids_minus_lm_vocab_size, 
                        uncond_input_ids, uncond_prefix,attention_mask, config, 
                        generator ,mask_token_id,mask_schedule,seq_len,temperature,step,sample_steps=steps)
                input_ids_t=global_id[step]
                input_ids_minus_lm_vocab_size_t=global_image[step]

                sampled_label,teacher_sampled_label,unknown_map_r=label_mask

                unknown_map_t,masking_pred, output_sampled,cond_logits = denoise(model,input_ids_t, input_ids_minus_lm_vocab_size_t,uncond_input_ids, uncond_prefix,
                    attention_mask, config,mask_token_id,seq_len,generator,mask_schedule,1,0,return_logits=True)
                unknown_map_t = unknown_map_t.reshape(-1)

                loss1=F.cross_entropy(cond_logits,sampled_label,reduction='none')

                loss_mask=unknown_map_r
                loss1=loss1*loss_mask
                loss1=loss1.sum()/loss_mask.sum()



                loss2=F.cross_entropy(cond_logits,teacher_sampled_label,reduction='none')
                loss2=loss2*unknown_map_t
                loss2=loss2.sum()/unknown_map_t.sum()

                # print("loss1:",loss1)
                # print("loss2:",loss2)
                if unknown_map_r.sum()>0:
                    loss3=0.05*loss1+loss2
                else:
                    loss3=loss2


                
                texts_lm = batch_lm["text"]
                input_ids_lm, _, labels_lm = uni_prompting((texts_lm, input_ids.shape[-1]), 'lm')
                attention_mask_lm = create_attention_mask_predict_next(input_ids_lm.to(model.device),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                attention_mask_lm = attention_mask_lm.to(mask_type)
                logits,loss_lm = calculate_lm_loss(
                    model=model,
                    input_ids=input_ids_lm,
                    input_embeddings=None,
                    attention_mask=attention_mask_lm,
                    labels=labels_lm,
                    label_smoothing=0.0,
                    max_seq_length=128,
                )


                loss=loss3+0.1*loss_lm+0.5*loss_mmu
                # accelerator.print(loss_lm)
                if accelerator.is_main_process:
                    step_value = step.item()
                    if step_value not in step_losses:
                        step_losses[step_value] = []
                        num_pred_dict[step_value] = []
                        num_hope_dict[step_value] = []
                        acc_dict[step_value] = []


                    step_losses[step_value].append((global_step, loss3.item(), loss_lm.item(), loss_mmu.item()))
                
                torch.cuda.empty_cache()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # print(model.module.dtype)
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # os.system("gpustat")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # 20.4.15. Make EMA update to target student model parameters
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:

                        if global_step % args.checkpointing_steps == 0:
                            # sample_prompt = 'a dog playing in the snow'
                            sample_prompt ="Two vespas parked next to a light post."
                            
                            sample_and_save_image(uni_prompting,global_step, sample_prompt, vq_model, model, mask_schedule, mask_token_id,sample_steps=4)

                            save_dir1=args.image_dir+ f'/loss_per_step_{global_step}'
                            if save_dir1 is not None:
                                os.makedirs(save_dir1, exist_ok=True)
                            # save_dir2=args.image_dir+ f'/num_pred_vs_num_hope_{global_step}'
                            # if save_dir2 is not None:
                            #     os.makedirs(save_dir2, exist_ok=True)

                            loss_jsonl_path = os.path.join(save_dir1, "loss_data.jsonl")
                            # pred_hope_jsonl_path = os.path.join(save_dir2, "pred_hope_data.jsonl")


                            for step_value in sorted(step_losses.keys()):
                                # Extract iterations and values for loss3
                                iterations_loss = [item[0] for item in step_losses[step_value]]
                                loss_values = [item[1] for item in step_losses[step_value]]
                                loss_lm_values = [item[2] for item in step_losses[step_value]]
                                loss_mmu_values = [item[3] for item in step_losses[step_value]]
                                

                                with open(loss_jsonl_path, 'a') as f:
                                    json.dump({"step": step_value, "iterations": iterations_loss, "losses": loss_values}, f)  # Save all at once
                                    f.write('\n')

                                # Plot loss3 over iterations for this step
                                plt.figure()
                                plt.plot(iterations_loss, loss_values, label='Loss3')   
                                plt.xlabel('Iteration')
                                plt.ylabel('Loss')
                                plt.title(f'Losses over Iterations for Step {step_value}')
                                plt.legend()  # Add legend to distinguish the different losses
                                plt.savefig(save_dir1 + f'/loss_per_step_{step_value}.png')
                                plt.close()

                                plt.figure()
                                plt.plot(iterations_loss, loss_lm_values, label='Loss_lm')
                                plt.xlabel('Iteration')
                                plt.ylabel('Loss')
                                plt.title(f'Loss_lm over Iterations for Step {step_value}')
                                plt.legend()
                                plt.savefig(save_dir1 + f'/loss_lm_per_step_{step_value}.png')
                                plt.close()

                                plt.figure()
                                plt.plot(iterations_loss, loss_mmu_values, label='Loss_mmu')
                                plt.xlabel('Iteration')
                                plt.ylabel('Loss')
                                plt.title(f'Loss_mmu over Iterations for Step {step_value}')
                                plt.legend()
                                plt.savefig(save_dir1 + f'/loss_mmu_per_step_{step_value}.png')
                                plt.close()


                                # iterations_acc = [item[0] for item in acc_dict[step_value]]
                                # acc_unknown_values = [item[1] for item in acc_dict[step_value]]
                                # acc_known_values = [item[2] for item in acc_dict[step_value]]
                                # plt.figure()
                                # plt.plot(iterations_acc, acc_unknown_values, label='acc_unknown')
                                # plt.plot(iterations_acc, acc_known_values, label='acc_known')
                                # plt.xlabel('Iteration')
                                # plt.ylabel('Accuracy')
                                # plt.title(f'acc_unknown vs acc_known over Iterations for Step {step_value}')
                                # plt.legend()
                                # plt.savefig(save_dir2+ f'/acc_unknown_vs_acc_known_{step_value}.png')
                                # plt.close()

                            
                            # teacher_sample_and_save_image(uni_prompting,global_step, sample_prompt, vq_model, teacher_model, mask_schedule,mask_token_id, sample_steps=16)
                            # del vq_model
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                        # if global_step % args.validation_steps == 0:

                        #     log_validation(vae, unet, args, accelerator, weight_dtype, global_step, "online")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    print("global_step",global_step)    
                    print("args.max_train_steps",args.max_train_steps)
                    break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(os.path.join(args.output_dir, "model"))

    print('training finished')
    accelerator.end_training()


if __name__ == "__main__":
    main(args)
