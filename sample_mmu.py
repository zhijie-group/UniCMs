import json
from tqdm import tqdm
import random
import argparse
import os
import sys
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import PIL
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, get_mask_chedule

from transformers import AutoTokenizer
import torch.nn.functional as F
from omegaconf import OmegaConf

from llava.llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28

IGNORE_INDEX = -100
# EOT_TOKEN = "<|EOT|>"




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


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def clean_prompt(prompt):
    # Remove line breaks
    cleaned_prompt = prompt.replace('\n', ' ')
    # Remove <image> tag
    cleaned_prompt = cleaned_prompt.replace('<image>', '')
    return cleaned_prompt


def preprocess_data(image_path, prompt, tokenizer, vq_model, uni_prompting, device, config):
    """
    Preprocesses a single image and prompt for the Show-o model.

    Args:
        image_path (str): Path to the image file.
        prompt (str): Text prompt.
        tokenizer: Tokenizer for text.
        vq_model: VQ-VAE model for image tokenization.
        uni_prompting: UniversalPrompting instance.
        device: Torch device.
        config: Configuration object.

    Returns:
        dict: Preprocessed data dictionary.
    """

    try:
        image_ori = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError, PIL.UnidentifiedImageError) as e:
        raise Exception(f"Error loading image {image_path}: {e}")

    image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
    image = image.unsqueeze(0)

    image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    batch_size = 1

    if len(prompt) > 1024:
        raise Exception("Prompt too long.")

    prompt = clean_prompt(prompt)
    input_text = ['USER: \n' + prompt + ' ASSISTANT:']

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

    return dict(sources_input_ids=input_ids, sources_len=[
        input.ne(tokenizer.pad_token_id).sum().item() for input in input_ids])


####### Get jacobian trajectory #######
@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    ):
    """
    Generates text using Jacobian trajectory sampling.

    Args:
        model: Show-o model.
        tokenizer: Tokenizer.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask.
        max_new_tokens (int): Maximum new tokens to generate per iteration.

    Returns:
        tuple: Trajectory IDs, last logits trajectory, eos_reached flag, iteration count.
    """

    bsz = input_ids.shape[0]
    prompt_len = [input_ids[i].shape[0] for i in range(bsz)]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device=device)

    for i in range(bsz):
        max_index = len(uni_prompting.text_tokenizer) - 1
        filtered_choices = [x for x in input_ids[i] if 0 <= x <= max_index]
        tokens[i, :] = torch.tensor(random.choices(filtered_choices, k=total_len)).to(dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")

    trajectory = []
    logits_trajectory = []
    next_generation = tokens
    next_text = tokenizer.decode(next_generation.tolist()[0])

    generate_attention_mask = create_attention_mask_for_mmu(next_generation.to(device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
    trajectory.append(tokens)
    itr = 0

    while True:
        current_generation = next_generation
        logits = model(current_generation, attention_mask=generate_attention_mask)

        logits_trajectory.append(logits)
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)

        # keep prompt unchanged, update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i] - 1:total_len - 1]), dim=0)

        trajectory.append(next_generation)
        itr += 1

        if torch.all(torch.eq(next_generation, current_generation)).item():
            eos_idxs = torch.where(trajectory[-1][0] == tokenizer.eos_token_id)
            eos_reached = len(eos_idxs[0]) > 1
            return trajectory[:-1], logits_trajectory[-1], eos_reached, itr


def main(image_path, prompt, output_file, max_new_tokens, max_new_seq_len,model_path,config):
    """
    Main function to perform Jacobi trajectory sampling and save the generated text.

    Args:
        image_path (str): Path to the input image.
        prompt (str): Text prompt.
        output_file (str): Path to save the output text file.
        max_new_tokens (int): Maximum new tokens per Jacobi iteration.
        max_new_seq_len (int): Maximum total sequence length.
    """
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(model_path)  # model path, you might need to change this path

    model = model.to(device)
    model.eval()
    print(model.showo.model.embed_tokens.weight.dtype)

    try:
        train_dataset = preprocess_data(image_path, prompt, tokenizer, vq_model, uni_prompting, device, config)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    inputs = torch.Tensor(train_dataset['sources_input_ids']).to(device=model.device, dtype=torch.int)
    attention_mask = create_attention_mask_for_mmu(inputs.to(device), eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

    itr = 0
    eos_reached = False
    jacobian_trajectory_ids = None  # Initialize to None for scope

    while itr * max_new_tokens < max_new_seq_len and not eos_reached:
        print('Retrieving one Jacobian trajectory...')
        jacobian_trajectory_ids, teacher_logits, eos_reached, iitr = get_jacobian_trajectory(model, tokenizer, inputs, attention_mask, max_new_tokens)
        itr += 1
        print(f'Jacobi iteration: {itr}')
        inputs = jacobian_trajectory_ids[-1]
        if eos_reached:
            print("EOS reached.")
            break

    if jacobian_trajectory_ids is not None:  # Check if trajectory was generated
        answer = jacobian_trajectory_ids[-1][0][train_dataset['sources_input_ids'][0].shape[0]:].tolist()
        first_eos_index = next((i for i, token in enumerate(answer) if token == tokenizer.eos_token_id), len(answer))
        fil_answer = answer[:first_eos_index]
        generated_text = tokenizer.decode(fil_answer)
        print(f'Generated text (Jacobi): {generated_text}')

        with open(output_file, 'w') as f:
            f.write(generated_text)
        print(f"Generated text saved to {output_file}")
    else:
        print("No text generated due to error or early termination.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--output_file", type=str, default="jacobi_output.txt",
                        help="Path to save the output text file")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="Maximum new tokens per Jacobi iteration")
    parser.add_argument("--max_new_seq_len", type=int, default=512,
                        help="Maximum total sequence length")
    parser.add_argument("--model_path",type=str,help="the UniCMs model path")
    parser.add_argument("--config_path",type=str,help="the Show-o config file, specially, the magvitv2 model load from this config file ")
    args = parser.parse_args()
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    config=OmegaConf.load(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print(uni_prompting.text_tokenizer.eos_token_id)
    print(uni_prompting.text_tokenizer.bos_token_id)

    main(args.image_path, args.prompt, args.output_file, args.max_new_tokens, args.max_new_seq_len,model_path=args.model_path,config=config)