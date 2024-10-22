import torch 
import numpy as np 
import evaluate 
import torch.multiprocessing as mp 
MODEL_PATH = '/nas/sunxiaofeng/models/llamafactory/tiny-random-Llama-3'
DATASET_PATH = '/nas/sunxiaofeng/LLaMA-Factory/data/sample_long_sft_32k_48M.json'
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator 
from easy_context import apply_lss_transformer_attn_monkey_patch_llama, apply_llama3_flash_attn_attn_monkey_patch_llama
accelerator = Accelerator()
MAX_LENGTH = 2048
def origin_forward(model, dataset, tokenizer):
    # 将数据集中的输入转换为模型可接受的格式
    rank = accelerator.state.process_index 
    inputs = tokenizer(dataset['train']['instruction'][0], return_tensors='pt', padding=True, truncation=True,max_length=MAX_LENGTH).to(f"{accelerator.device}")
    inputs['attention_mask'] = None
    # inputs = {k: v.to(f"{accelerator.device}:0") for k, v in inputs.items()}
    model = model.to(f"{accelerator.device}")
    position_ids = torch.arange(0, MAX_LENGTH, dtype=torch.long, device=accelerator.device).unsqueeze(0).expand(1, -1)
    # 使用模型进行前向计算
    
    outputs = model(**inputs,output_hidden_states=True,position_ids=position_ids,past_key_values=None)
    # 返回模型的输出
    origin_result = outputs.logits
    intermediate_results = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None 
    return origin_result.squeeze(0), intermediate_results

def sp_forward(model, dataset, tokenizer,algorithm_name=None):
    device = accelerator.device
    rank = accelerator.state.process_index 
    num_process = accelerator.state.num_processes
    inputs = tokenizer(dataset['train']['instruction'][0], return_tensors='pt', padding=True, truncation=True,max_length=MAX_LENGTH)
    local_seq_length = int(MAX_LENGTH / num_process)
    inputs['input_ids'] = inputs['input_ids'][:, local_seq_length * rank: local_seq_length * (rank + 1)].to(f"{accelerator.device}")
    # inputs['attention_mask'] = inputs['attention_mask'][:, local_seq_length * rank: local_seq_length * (rank + 1)].to(f"{accelerator.device}")
    inputs['attention_mask'] = None
    model = model.to(f"{accelerator.device}")
    position_ids = torch.arange(0, MAX_LENGTH, dtype=torch.long, device=accelerator.device).unsqueeze(0).expand(1, -1)
    if algorithm_name == "lss_transformer":
        apply_lss_transformer_attn_monkey_patch_llama()
    elif algorithm_name == 'llama3_flash_attn':
        apply_llama3_flash_attn_attn_monkey_patch_llama()

    if algorithm_name == "lss_transformer":    
        outputs = model(**inputs,output_hidden_states=True,position_ids=position_ids,past_key_values=None)
    elif algorithm_name == 'llama3_flash_attn':
        outputs = model(**inputs,output_hidden_states=True,position_ids=position_ids[:,local_seq_length*rank:local_seq_length*(rank+1)],past_key_values=None)
    sp_result = outputs.logits
    intermediate_results = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None 
    return sp_result.squeeze(0), intermediate_results

def main(algorithm_name = None):
    # load model and prepare data
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        pad_token = '<pad>'
    )
    dataset = load_dataset('json',data_files = DATASET_PATH)
    if algorithm_name != 'llama3_flash_attn':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    else: 
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    rank = accelerator.state.process_index
    # origin model forward
    if rank == 0:
        origin_result, origin_inter = origin_forward(model,dataset,tokenizer) 
    
    # sp model forward
    sp_result, sp_inters = sp_forward(model,dataset,tokenizer,algorithm_name=algorithm_name)
    all_sp_results = accelerator.gather(sp_result) 
    all_sp_inter = []
    for sp_inter in sp_inters:
        sp_inter = sp_inter.squeeze(0)
        all_sp_inter.append(accelerator.gather(sp_inter))
    if rank == 0: 
        for idx, inter in enumerate(all_sp_inter):
            print(f"{idx}th decoder result is the same ? {torch.allclose(origin_inter[idx], inter)}")
            assert torch.allclose(origin_inter[idx], inter)
        print("-------------------VERIFIED PASSED-------------------")
        # assert torch.allclose(origin_result, all_sp_results,rtol=1e-5) 
        # LLamaForCausalLM的logits，不符合SP
        

if __name__ == "__main__":
    main('llama3_flash_attn')