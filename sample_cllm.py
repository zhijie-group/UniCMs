import json
from tqdm import tqdm
import random
import argparse
import os
import sys
os.chdir('/home/chenkai/data/mcm_showo')
sys.path.append('/home/chenkai/data/mcm_showo')
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftModel
print(os.getcwd())
print(sys.path)
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

config_file_path = '/home/chenkai/data/Show-o/configs/showo_demo.yaml'
config = OmegaConf.load(config_file_path)
image_prefix="/home/chenkai/data/g3_dataset/llava_train_image/val2014/COCO_val2014_"

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

def jacobian_generated_data_postprocessed(generated_data, tokenizer):
    low_quality_data_id_lst = []
    # delete low quality data with repetitive pattern
    for i, d in enumerate(generated_data):
        if detect_repetitive_patterns(tokenizer, np.array(d['prompt_ids']), repeat_ngram_size=10):
            prompt_ids = np.array(d['prompt_ids'])
            if len(prompt_ids.shape)==2:
                prompt_ids = prompt_ids[0]
            elif len(prompt_ids.shape)==3:
                prompt_ids = prompt_ids[0][0]
            print(f'Low quality generation detected: {tokenizer.decode(prompt_ids)}')
            low_quality_data_id_lst.append(i)
    print(f'{len(low_quality_data_id_lst)} low quality data detected. {len(low_quality_data_id_lst)/len(generated_data)} percent of low quality data.')

    # add complete teacher outputs
    teacher_output_inspector = {}
    for d in generated_data:
        data_id = d["data_id"]
        if data_id in teacher_output_inspector.keys():
            all_teacher_output_map = teacher_output_inspector[data_id]
        else:
            all_teacher_output_map = {}
            #print(data_id)
        itr = d["jacobian_itr_id"]
        # handle bsz=1 case only
        all_teacher_output_map[itr] = d["teacher_output_ids"][0]
        teacher_output_inspector[data_id] = all_teacher_output_map

    teacher_output_collector = {}
    for d_id in teacher_output_inspector.keys():
        all_teacher_output_map = teacher_output_inspector[d_id]
        all_itr = [int(s.split('_')[1]) for s in all_teacher_output_map.keys()]
        # print(all_itr)
        max_itr = max(all_itr)
        max_itr_s = "itr_" + str(max_itr)
        complete_teacher_output = all_teacher_output_map[max_itr_s]
        teacher_output_collector[d_id] = complete_teacher_output

    f_result = []
    for d in generated_data:
        data_id = d["data_id"]
        complete_teacher_output = teacher_output_collector[data_id]
        d["complete_teacher_output_ids"] = complete_teacher_output
        f_result.append(d)
    
    cleaned_f_result = []
    for i, d in enumerate(generated_data):
        if i in low_quality_data_id_lst:
            continue
        cleaned_f_result.append(d)


    return cleaned_f_result


def clean_prompt(prompt):
    # 移除换行符
    cleaned_prompt = prompt.replace('\n', ' ')
    # 移除 <image>
    cleaned_prompt = cleaned_prompt.replace('<image>', '')
    return cleaned_prompt

def load_json_lines(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))  # 逐行解析JSON
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")  # 错误处理
    return data


def mmu_generate(model,idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    try:
        device = idx.device
    except:
        device = input_embeddings.device

    result = []
    num=0
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        # logits, _ = self(idx_cond)
        logits = model(idx, input_embeddings=input_embeddings, attention_mask=attention_mask,use_cache=True)

        L = attention_mask.shape[-1]
        attention_mask = attention_mask.squeeze()
        attention_mask_a = torch.hstack(
            [
                attention_mask,  # L, L
                torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
            ]
        )
        attention_mask_b = torch.vstack(
            [
                attention_mask_a,  # L, L+1
                torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
            ]
        )
        attention_mask = attention_mask_b

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        result.append(idx_next[0][0])
        # append sampled index to the running sequence and continue

        idx = torch.cat((idx, idx_next), dim=1)
        num+=1

        if eot_token is not None and idx_next.cpu() == eot_token:
            break
    print("num:",num)
    return result

def preprocess_data(data, tokenizer):
    train_dataset = []
    questions = ["What scene does this image depict?"]
    for i in tqdm(range(len(data))):
        # if i<60000:
        #     continue
        if i>0:
            break

        try:
            d = data[i]
            data_id = d["question_id"]
            prompt = d["text"]
            
            image_path = image_prefix + d["image"]
            prompt=questions[i]
            image_path="/home/chenkai/data/Show-o/mmu_validation/00000450_-4508645165214157965.jpg"
            

            try:
                image_ori = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, OSError, PIL.UnidentifiedImageError) as e:
                print(f"Error loading image {image_path}: {e}")
                continue  # Skip to the next data item

            image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
            image = image.unsqueeze(0)

            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            batch_size = 1

            if len(prompt) > 1024:
                print(f"Skipping data item {data_id}: Prompt too long.")
                continue

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


            train_dataset.append(dict(sources_input_ids=input_ids, sources_len=[
                input.ne(tokenizer.pad_token_id).sum().item() for input in input_ids], 
                                      image_path=image_path, data_id=data_id))
            
            # if i>1:
            #     break


        except Exception as e:  # Catch any other potential errors
            print(f"Error processing data item {i}: {e}")
            continue  # Skip to the next data item

    return train_dataset


####### Get jacobian trajectory #######
@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    output_file_path='/home/chenkai/data/mcm_showo/record.txt'  # 增加一个参数指定输出文件的路径
    ):

    bsz = input_ids.shape[0]
    prompt_len=[input_ids[i].shape[0] for i in range(bsz)]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # 打开文件写入模式（'w'）来写入数据
    with open(output_file_path, 'w') as f:
        # 记录一些初始的关键信息
        f.write(f"bsz: {bsz}, total_len: {total_len}\n")

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
        f.write(f"itr: {itr}\n")

        # 将初始的token解码并写入文件
        cut=25
        tokens0 = next_generation.tolist()[0]
        tokens0=tokens0[prompt_len[0]:prompt_len[0]+cut]
        decoded_tokens = [tokenizer.decode([token]) for token in tokens0]
        # 将解码后的特殊字符用可见的方式代表（仅在 token 中仅包含不可见字符时进行转换）
        decoded_tokens = [token if any(c.isprintable() and not c.isspace() for c in token) else token.replace(' ', '<space>').replace('\n', '<newline>') for token in decoded_tokens]
        decoded_text_with_spaces = " ".join(decoded_tokens)
        f.write(f"{decoded_text_with_spaces}\n")

        while True:
            current_generation = next_generation
            logits = model(current_generation, attention_mask=generate_attention_mask)

            logits_trajectory.append(logits)
            next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)

            # 保持 prompt 不变，更新生成的 tokens
            for i in range(bsz):
                next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)

            next_text = tokenizer.decode(next_generation.tolist()[0])
            trajectory.append(next_generation)
            itr += 1
            f.write(f"itr: {itr}\n")

            # 将每次迭代后的 token 解码并写入文件
            tokens0 = next_generation.tolist()[0]
            tokens0=tokens0[prompt_len[0]:prompt_len[0]+cut]
            decoded_tokens = [tokenizer.decode([token]) for token in tokens0]
            # 将解码后的特殊字符用可见的方式代表（仅在 token 中仅包含不可见字符时进行转换）
            decoded_tokens = [token if any(c.isprintable() and not c.isspace() for c in token) else token.replace(' ', '<space>').replace('\n', '<newline>') for token in decoded_tokens]
            decoded_text_with_spaces = " ".join(decoded_tokens)
            f.write(f"{decoded_text_with_spaces}\n")

            if torch.all(torch.eq(next_generation, current_generation)).item():
                eos_idxs = torch.where(trajectory[-1][0] == tokenizer.eos_token_id)
                eos_reached = len(eos_idxs[0]) > 1
                return trajectory[:-1], logits_trajectory[-1], eos_reached, itr

        

import time
import json
import os
from tqdm import tqdm

def main(filename, model, tokenizer, max_new_tokens, max_new_seq_len, use_aug, use_labels, data_size):
    data = load_json_lines(filename)
    train_dataset = preprocess_data(data, tokenizer)

    prompt_size = len(train_dataset)
    counter = 0
    counter_itr = 0
    new_data = []
    all_text = []
    total_time_jacobian = 0
    total_time_ar = 0
    total_tokens_jacobian = 0
    total_tokens_ar = 0

    for i in tqdm(range(prompt_size)):
        d = train_dataset[i]
        inputs = torch.Tensor(d['sources_input_ids']).to(device=model.device, dtype=torch.int)
        attention_mask = create_attention_mask_for_mmu(inputs.to(device), eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
        
        itr = 0
        total_itr = 0
        eos_reached = False

        # Jacobian method
        start_time_jacobian = time.time()
        while itr * max_new_tokens < max_new_seq_len and not eos_reached:
            dic = {'data_id': d['data_id'], 'jacobian_itr_id': f'itr_{itr}', 'prompt_ids_len': d['sources_len']}
            print('retrieving one Jacobian trajectory...')
            jacobian_trajectory_ids, teacher_logits, eos_reached, iitr = get_jacobian_trajectory(model, tokenizer, inputs, attention_mask, max_new_tokens)
            total_itr += iitr
            
            dic["answer_trajectory_ids"] = [id[0][-max_new_tokens:].tolist() for id in jacobian_trajectory_ids]
            inputs = jacobian_trajectory_ids[-1]
            dic['teacher_output_ids'] = jacobian_trajectory_ids[-1].tolist()
            new_data.append(dic)
            itr += 1
            
            print(f'writing counter = {counter}...')
            counter += 1
            counter_itr += iitr
            break

        answer = jacobian_trajectory_ids[-1][0][d['sources_input_ids'][0].shape[0]:].tolist()
        # Calculate tokens before the first EOS for Jacobian
        first_eos_index = next((i for i, token in enumerate(answer) if token == tokenizer.eos_token_id), len(answer))
        total_tokens_jacobian += first_eos_index
        fil_answer=answer[:first_eos_index]
        # print("fil_answer_text:",tokenizer.decode(fil_answer))
        
        end_time_jacobian = time.time()
        total_time_jacobian += end_time_jacobian - start_time_jacobian


        # Autoregressive method
        # inputs_ar = torch.Tensor(d['sources_input_ids']).to(device=model.device, dtype=torch.int)
        # attention_mask_ar = create_attention_mask_for_mmu(inputs_ar.to(device), eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
        
        # start_time_ar = time.time()
        # cont_toks_list = mmu_generate(model, inputs_ar, attention_mask=attention_mask_ar, max_new_tokens=512, top_k=1, eot_token=uni_prompting.sptids_dict['<|eot|>'])
        # cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        # # Calculate tokens before the first EOS for AR
        # first_eos_index_ar = next((i for i, token in enumerate(cont_toks_list[0]) if token == tokenizer.eos_token_id), len(cont_toks_list[0]))
        # total_tokens_ar += first_eos_index_ar
        # fil_answer_ar=cont_toks_list[0][:first_eos_index_ar]
        # # print("fil_answer_ar_text:",tokenizer.decode(fil_answer_ar))
        
        # end_time_ar = time.time()
        # total_time_ar += end_time_ar - start_time_ar

        
        text = tokenizer.batch_decode([answer], skip_special_tokens=False)
        # text_ar = tokenizer.batch_decode(cont_toks_list, skip_special_tokens=False)
        all_text.append(text)
        print(f'Generated text jacobi: {text}')
        # print(f'Generated text ar: {text_ar}')
        print("total_itr:",total_itr)   
        print("avg_itr:",total_itr/itr)
    
    # Summary of results
    total_avg_speed_jacobian = total_time_jacobian / total_tokens_jacobian if total_tokens_jacobian > 0 else float('inf')
    # total_avg_speed_ar = total_time_ar / total_tokens_ar if total_tokens_ar > 0 else float('inf')

    print('Jacobi trajectory has been collected. Now delete low-quality generation as post processing.')
    print(f'Total time for Jacobian: {total_time_jacobian}, Total tokens: {total_tokens_jacobian}, Avg speed: {total_avg_speed_jacobian} sec/tokens')
    # print(f'Total time for AR: {total_time_ar}, Total tokens: {total_tokens_ar}, Avg speed: {total_avg_speed_ar} sec/tokens')

    save_file = os.path.join(ck_path, "itr_record.json")
    with open(save_file, 'w') as f_merged:
        json.dump({"counter": counter, "counter_itr": counter_itr, "total_avg_itr": counter_itr / counter if counter > 0 else 0}, f_merged)
        json.dump(all_text, f_merged)




    # save_path = '/home/xck/data/Consistency_LLM-main/'    
    # # save_path=
    # cleaned_data = jacobian_generated_data_postprocessed(new_data, tokenizer)
    # new_file_name = "cleaned_" + f"jacobi_max_new_tokens{max_new_tokens}_aug{use_aug}_labels_{use_labels}_max_seq_len_{max_new_seq_len}.json"
    # new_file_path = os.path.join(save_path, new_file_name)
    
    # # create directory for a path if it doesn't exist
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # with open(new_file_path, 'w') as f_merged:
    #     json.dump(cleaned_data, f_merged)
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,
                        default="/home/chenkai/data/g3_dataset/llava_instruct_150k/qa90_questions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str,
                        default="models/vicuna-7b-v1.5")
    parser.add_argument("--data_size", default=5000)
    parser.add_argument("--use_aug", default=True)
    parser.add_argument("--use_labels", default=True)
    args = parser.parse_args()
    filename = args.filename
    # model_path = config.model.showo.pretrained_model_path
    

    max_new_tokens = args.max_new_tokens
    max_new_seq_len = args.max_new_seq_len

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    
    print(uni_prompting.text_tokenizer.eos_token_id)
    print(uni_prompting.text_tokenizer.bos_token_id)
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # vision_tower_name = "openai/clip-vit-large-patch14-336"
    # vision_tower =  CLIPVisionTower(vision_tower_name).to(device)
    # clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    # ck_path="/home/chenkai/data/g3_data/showo_mcllm_final_noreg_lr1e-5/checkpoint-1000"
    # ck_path="/home/chenkai/data/ckpt/showo_mcllm_16-1_lr1e-5/checkpoint-5500"
    # ck_path="/home/xck/data/ckpt/showo_mcllm_1t2i05mmu_nolora_lr1e-6/checkpoint-34000"
    # ck_path=config.model.showo.pretrained_model_path
    # ck_path="/home/xck/data/ckpt/showo_mcllm_1t2i05mmu_lora_phase2_8-2_lr1e-6/checkpoint-5"
    model = Showo.from_pretrained("/home/chenkai/data/g3_data/showo_mcllm_final8-2_phase2_lr1e-5/checkpoint-27000")
    # print(model)


    # logging_dir = Path(args.output_dir, args.logging_dir)

    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # ck_path="/home/chenkai/data/ckpt/showo_mcllm_finallora_lr1e-4/checkpoint-2000/model"
    # # input_dir=os.path.join(ck_path, "model")
    # input_dir=ck_path
    # peft_config = LoraConfig.from_pretrained(input_dir)
    # model = PeftModel.from_pretrained(model,input_dir, config=peft_config) 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42) 
    model = model.to(device)  
    model.eval()
    print(model.showo.model.embed_tokens.weight.dtype)

    
    main(filename, model, tokenizer, max_new_tokens, max_new_seq_len, args.use_aug, args.use_labels, args.data_size)
