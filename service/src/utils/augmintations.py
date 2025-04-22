from typing import Callable

class BaseAugmintation:
    def __init__(self, f: Callable):
        self.func = f
    def __call__(self, x):
        return x