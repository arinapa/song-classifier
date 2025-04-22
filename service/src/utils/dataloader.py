import random
from typing import Any


class BaseDataloader:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self,):
        return len(self.data)
    
class DataWrapper:
    def __init__(self, dataloader: BaseDataloader):
        self.dl = dataloader
    def __call__(self, n: int|None = None): #TODO batch_size and augs support
        if n is None:
            indexes = list(range(len(self.dl)))
        indexes = random.choices(range(len(self.dl)), k=n)
        return [self.dl[idx] for idx in indexes]
