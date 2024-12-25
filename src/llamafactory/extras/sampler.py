import torch
from torch.utils.data.sampler import Sampler
import random
from typing import (
    Iterator,
    Sized,
)


class DynamicBatchSampler(Sampler[int]):
    
    data_source: Sized

    def __init__(self, data_source: Sized, max_seq_length: int, shuffle: bool=True)-> None:
        self.data_source = data_source
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.indices = list(range(len(data_source)))
    
    def __iter__(self)-> Iterator[int]:
        if self.shuffle:
            random.shuffle(self.indices)

        batch = []
        current_length = 0

        for idx in self.indices:
            seq_length = len(self.data_source[idx]["input_ids"])  # 获取序列长度
            if current_length + seq_length <= self.max_seq_length:
                batch.append(idx)
                current_length += seq_length
            else:
                yield batch
                batch = [idx]
                current_length = seq_length
        
        if batch:
            yield batch
    
    def __len__(self):
        return len(self.data_source)


def new__iter__(self, train_batch_size=None, dp_size=None) -> Iterator[int]:
    n = len(self.data_source)
    if self.generator is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = self.generator

    if self.replacement:
        for _ in range(self.num_samples // 32):
            yield from torch.randint(
                high=n, size=(32,), dtype=torch.int64, generator=generator
            ).tolist()
        yield from torch.randint(
            high=n,
            size=(self.num_samples % 32,),
            dtype=torch.int64,
            generator=generator,
        ).tolist()
    else:
        for _ in range(self.num_samples // n):
            # yield from torch.randperm(n, generator=generator).tolist()
            yield from rerank(self.data_source, self.generator, n, train_batch_size, dp_size)

        yield from torch.randperm(n, generator=generator).tolist()[
            : self.num_samples % n
        ]

def rerank(data_source, generator, n, train_batch_size, dp_size):
    perm = torch.randperm(n, generator=generator).tolist()
    result = []
    for i in range(0, n, train_batch_size):
        if i+train_batch_size>n+1:break
        batch_idx = perm[i:i+train_batch_size]
        # 将batch按顺序排列后Z字形逻辑取
        batch_tokens = [len(data_source[idx]['input_ids']) for idx in batch_idx]
        sorted_indices = sorted(range(len(batch_tokens)), key=lambda i: batch_tokens[i])
        sorted_batch_idx = [batch_idx[item] for item in sorted_indices]
        result.extend(group_indices(sorted_batch_idx, dp_size))
        # print(f"batch_idx:{batch_idx}, batch_tokens:{batch_tokens}, sorted_indices:{sorted_indices}, sorted_batch_idx:{sorted_batch_idx}")
        # print(f"group_indices:{group_indices(sorted_batch_idx, dp_size)}")
    return result

def group_indices(lst, N):
    # 初始化每组的索引列表
    groups = [[] for _ in range(N)]
    # 计算每个块的大小
    chunk_size = N
    # 遍历列表，按块处理
    for chunk_start in range(0, len(lst), chunk_size):
        # 当前块的索引范围
        chunk_indices = list(range(chunk_start, chunk_start + chunk_size))
        # 计算当前块的组分配顺序
        if (chunk_start // chunk_size) % 2 == 0:
            # 顺序分配
            for i in range(N):
                groups[i].append(lst[chunk_indices[i]])
        else:
            # 逆序分配
            for i in range(N):
                groups[i].append(lst[chunk_indices[N - 1 - i]])
    return [item for group in groups for item in group]