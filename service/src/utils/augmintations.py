from typing import Callable

class BaseAugmintation: # посмотреть аугментации в торче
    def __init__(self, f: Callable):
        self.func = f
    def __call__(self, x):
        return x