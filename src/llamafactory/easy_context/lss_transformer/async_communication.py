import threading
import math
import os

import torch
import torch.distributed as dist
from torch.distributed import batch_isend_irecv, P2POp, isend, irecv, get_process_group_ranks


def initialize_distributed(sp_size=None):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
    else:
        # 初始化dist设置
        if int(os.environ["RANK"]) == 0:
            print("Initializing Torch distributed.")
        dist.init_process_group(backend="nccl")
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        global_world_size = dist.get_world_size()
        torch.cuda.set_device(dist.get_rank() % local_world_size)
