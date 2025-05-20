import torch
from tqdm.auto import tqdm
import argparse
import logging
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next


torch.cuda.empty_cache()


def inference(config_file_path, ck_path, prompt, inference_step, guidance_scale, device_num=0, is_lora=False, top_k=None, output_path="output.png"):
    """
    Generates an image based on the given prompt and saves it.

    Args:
        config_file_path (str): Path to the configuration file.
        ck_path (str): Path to the model checkpoint.
        prompt (str): Text prompt for image generation.
        inference_step (int): Number of inference steps.
        guidance_scale (float): Guidance scale.
        device_num (int): CUDA device number to use, default is 0.
        is_lora (bool): Whether to use LoRA model, default is False.
        top_k (int, optional): Top-k sampling parameter, default is None.
        output_path (str): Path to save the generated image, default is "output.png".
    """

    from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from PIL import Image
    import numpy as np
    import torch
    from models import Showo, MAGVITv2, get_mask_chedule
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    import torch
    import numpy as np
    from PIL import Image

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

    def denoise(model, input_ids, input_ids_minus_lm_vocab_size, uncond_input_ids, uncond_prefix, attention_mask, config, generator, ratio, mask_token_id, noise_schedule, seq_len, temperature):
        if uncond_input_ids is not None and config.training.guidance_scale > 0:
            uncond_input_ids = torch.cat(
                [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            cond_logits, uncond_logits = forward(model, model_input, attention_mask=attention_mask).chunk(2)
            logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
            logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
        else:
            logits = forward(model, input_ids, attention_mask=attention_mask)
            logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]

        probs = logits.softmax(dim=-1)
        sampled = probs.reshape(-1, logits.size(-1))

        if top_k is not None:
            topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)
            topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
            sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]
            sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])

        else:
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

        unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
        sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

        mask_ratio = noise_schedule(torch.tensor(ratio))
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
        selected_probs = selected_probs.squeeze(-1)
        selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

        mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
        mask_len = torch.max(
            torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
        )
        temperature = temperature * (1.0 - ratio)
        masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

        input_ids[:, -(seq_len + 1):-1] = torch.where(masking, mask_token_id,
                                                        sampled_ids + config.model.showo.llm_vocab_size + 10)
        input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids

    os.environ["WANDB_MODE"] = "offline"

    def get_vq_model_class(model_type):
        if model_type == "magvitv2":
            return MAGVITv2
        else:
            raise ValueError(f"model_type {model_type} not supported.")

    torch.cuda.empty_cache()

    config = OmegaConf.load(config_file_path)
    config.mode = 't2i'

    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    if is_lora:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        input_dir = ck_path
        peft_config = LoraConfig.from_pretrained(input_dir)
        model = PeftModel.from_pretrained(model, input_dir, config=peft_config)
        model = model.to(device)
        model.eval()
    else:
        input_dir = ck_path
        model = Showo.from_pretrained(input_dir).to(device)
        model.eval()

    dtype = torch.float16
    model = model.to(dtype=dtype)
    vq_model = vq_model.to(dtype=dtype)

    config.training.batch_size = 1 # batch size is usually 1 for inference
    config.training.guidance_scale = guidance_scale
    config.training.generation_timesteps = inference_step

    mask_token_id = model.config.mask_token_id

    image_dir = os.path.dirname(output_path)
    if not os.path.exists(image_dir) and image_dir != '': # create directory if output path's directory does not exist
        os.makedirs(image_dir)

    image_tokens = torch.ones((1, config.model.showo.num_vq_tokens),
                                dtype=torch.long, device=device) * mask_token_id

    input_ids, _ = uni_prompting(([prompt], image_tokens), 't2i_gen')

    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * 1, image_tokens), 't2i_gen')
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
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    with torch.no_grad():
        seed = 0 # fix seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        temperature = config.training.get("generation_temperature", 1.0)
        noise_schedule = mask_schedule
        noise_type = config.training.get("noise_type", "mask")
        seq_len = config.model.showo.num_vq_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - 10)

        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]
        else:
            uncond_prefix = None


        for step in range(config.training.generation_timesteps):
            ratio = 1.0 * (step + 1) / config.training.generation_timesteps
            input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids = denoise(
                model, input_ids, input_ids_minus_lm_vocab_size,
                uncond_input_ids, uncond_prefix, attention_mask, config,
                generator, ratio, mask_token_id, noise_schedule, seq_len, temperature)

    gen_token_ids = sampled_ids
    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(output_path) # save generated image

    print(f"Image saved to: {output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-image inference using Show-o model.")
    parser.add_argument("--config_file_path", type=str, default='UniCMs/config/showo_512.yaml', help="path to config file")
    parser.add_argument("--ck_path", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="text prompt for image generation")
    parser.add_argument("--inference_step", type=int, default=20, help="number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--device_num", type=int, default=0, help="CUDA device number")
    parser.add_argument("--is_lora", action="store_true", help="whether to use LoRA model")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--output_path", type=str, default="output.png", help="image save path")

    args = parser.parse_args()

    inference(
        config_file_path=args.config_file_path,
        ck_path=args.ck_path,
        prompt=args.prompt,
        inference_step=args.inference_step,
        guidance_scale=args.guidance_scale,
        device_num=args.device_num,
        is_lora=args.is_lora,
        top_k=args.top_k,
        output_path=args.output_path
    )