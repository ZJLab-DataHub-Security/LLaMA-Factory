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