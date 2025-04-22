from typing import Any, Iterable
from service.src.utils.utils import open_file
from service.src.utils.augmintations import BaseAugmintation
from service.src.utils.dataloader import BaseDataloader, DataWrapper
from service.src.model.basemodel import BaseRecognitionModel


class CalcMetric:
    def __init__(self, dataloader: BaseDataloader, augs: Iterable[BaseAugmintation]|None = None):
        self.dl = dataloader
        self.augs = augs
        
    def __call__(self, model: BaseRecognitionModel):
        dw = DataWrapper(self.dl)
        result = []
        for song in dw(None):
            data = open_file(song.file)
            answer = model(data)
            result.append(0)
            if not answer is None:
                if answer.name == song.name:
                    result[-1]+=(0.5)
                if answer.artist == song.artist:
                    result[-1]+=(0.5)
        accuracy = sum(result)/len(result)
        return accuracy
        

        
