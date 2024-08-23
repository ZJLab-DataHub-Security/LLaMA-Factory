import os
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import linalg as LA
import torch
import torch.distributed as dist
from transformers import LlamaModel, LlamaConfig
import random
# print(__package__)
# __package__ = 'lss_transformer.test'
# from ..monkey_patch import apply_lss_transformer_attn_monkey_patch_llama
model_path = '/mnt/zj-gpfs/home/qianhao/models/Meta-Llama-3-8B'

def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)             
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 4096
    nheads = 32
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    X = torch.randn(batch_size, seqlen, nheads*d, device=device, dtype=dtype, requires_grad= False) # input: [b,s,h]
    dist.broadcast(X, src=0)

    local_X = X.chunk(world_size, dim=1)[rank]# [b, s/p, h]
    position_ids = torch.ones(batch_size, X.shape[1]).to(device=device)
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# origin forward:")
        print("#" * 30)


    # for forward
    # origin version
    configuration = LlamaConfig()
    origin_decoder = LlamaDecoderLayer(configuration,0).to(device=device)
    # if rank == 0 :
    #     print(origin_decoder)
    with torch.no_grad():
        origin_output_local = origin_decoder(X,position_ids=position_ids.clone())[0] 
    # 
    if rank == 0:
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
    lss_transformer.monkey_patch.apply_lss_transformer_attn_monkey_patch_llama()
    with torch.no_grad():
        lss_output_local = origin_decoder(local_X,position_ids = position_ids)[0] # 传递部分的X_i
    
    # all gather output
    all_lss_output = [torch.zeros_like(lss_output_local) for _ in range(world_size)]
    dist.all_gather(all_lss_output, origin_output_local)
    dist.barrier()
    all_lss_output = torch.cat(all_lss_output, dim=1) # s
    if rank == 0:
        print(LA.matrix_norm(all_lss_output))      

    dist.destroy_process_group()
    # for backward 

    # dist.barrier()
    # if rank == 0:
    #     print("#" * 30)
    #     print("# backward:")
    #     print("#" * 30)

    # out.backward(dout)
    # dqkv = qkv.grad
    # local_dqkv = dqkv.chunk(world_size, dim=1)[rank]

    # ring_out.backward(local_dout)
    # ring_dqkv = local_qkv.grad

    # log("local_dq", local_dqkv[:, :, 0, :])
    # log("dq diff", local_dqkv[:, :, 0, :] - ring_dqkv[:, :, 0, :])

    # log("local_dk", local_dqkv[:, :, 1, :])
    # log("dk diff", local_dqkv[:, :, 1, :] - ring_dqkv[:, :, 1, :])

    # log("local_dv", local_dqkv[:, :, 2, :])
    # log("dv diff", local_dqkv[:, :, 2, :] - ring_dqkv[:, :, 2, :])
