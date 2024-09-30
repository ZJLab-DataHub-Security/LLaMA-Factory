import os
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from torch import linalg as LA
import torch
import torch.distributed as dist
from transformers import LlamaModel, LlamaConfig
import random

def set_seed(seed=42):
    random.seed(seed)             
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 

def test_LlamaRMSNorm():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个随机矩阵d
    batch_size, seq_len, hidden_size = 2, 300, 768
    matrix = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 创建LlamaRMSNorm实例
    rms_norm = LlamaRMSNorm(hidden_size).to(device)
    
    # 对整个矩阵进行RMSNorm
    with torch.no_grad():
        full_norm = rms_norm(matrix.clone())
    
    # 对矩阵的前1/3进行RMSNorm
    partial_len = seq_len // 3
    partial_matrix = matrix[:, :partial_len, :]
    with torch.no_grad():
        partial_norm = rms_norm(partial_matrix)
    
    # 比较结果
    diff = torch.abs(full_norm[:, :partial_len, :] - partial_norm)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print(f"rms max_diff: {max_diff.item()}")
    print(f"rms mean_diff: {mean_diff.item()}")
    
    # 判断差异是否在可接受范围内（例如，小于1e-5）
    tolerance = 1e-5
    is_close = torch.allclose(full_norm[:, :partial_len, :], partial_norm)
    print(f"RMS is same ??: {is_close}")

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 4
    nheads = 4
    d = 4
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0

    # if rank == 0:
    #     print("#" * 30)
    #     print("# test LlamaRMSNorm:")
    #     print("#" * 30)
    #     test_LlamaRMSNorm()

    if dist.get_rank() == 0:
        # X = torch.randn(batch_size, seqlen, nheads*d, device=device, dtype=dtype, requires_grad=False)
        X = torch.arange(1, batch_size * seqlen * nheads * d + 1, dtype=dtype, device=device).reshape(batch_size, seqlen, nheads * d)
        # print(X)
    else:
        X = torch.empty(batch_size, seqlen, nheads*d, device=device, dtype=dtype, requires_grad=False)

    dist.broadcast(X, src=0)

    position_ids = torch.arange(0, X.shape[1], dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# origin forward:")
        print("#" * 30)

    # test LlamaRMSNorm

    # for forward
    # origin version
    configuration = LlamaConfig(hidden_size=16,num_attention_heads=4,_attn_implementation='sdpa')
    origin_decoder = LlamaDecoderLayer(configuration,0).to(device=device).eval()
    if rank == 0:
        with torch.no_grad(): 
            origin_output_local = origin_decoder(X,position_ids=position_ids.clone())[0] 
            print(LA.matrix_norm(origin_output_local))  # 完整的输出

    # lss
    if rank == 0:
        print("#" * 30)
        print("# lss forward:")
        print("#" * 30)
    import sys, os 
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0,parent_dir)
    try:
        import lss_transformer
    except Exception as e :
        print(e)
        sys.exit()
    dist.barrier()
    local_X = X.chunk(world_size, dim=1)[rank]# [b, s/p, h]
    lss_transformer.monkey_patch.apply_lss_transformer_attn_monkey_patch_llama()
    with torch.no_grad():
        lss_output_local = origin_decoder(local_X,position_ids = position_ids)[0] # 传递部分的X_i
    
    # all gather output
    all_lss_output = [torch.zeros_like(lss_output_local) for _ in range(world_size)]
    dist.all_gather(all_lss_output, lss_output_local)
    dist.barrier()
    all_lss_output = torch.cat(all_lss_output, dim=1) 
    if rank == 0:
        print(LA.matrix_norm(all_lss_output))      
        print(f"the output of origin and lss is same??: {torch.allclose(origin_output_local, all_lss_output)}")
    dist.destroy_process_group()