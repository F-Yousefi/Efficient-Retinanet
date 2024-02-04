from typing import Any
from utils.config import config
from torch.utils import data
import torch
import torchvision


class Dataset(data.Dataset):
    def __init__(self, dataset, transform = None, fraction = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = config.device
        self.fraction = fraction
        self.transform = transform

    def __getitem__(self, index) -> Any:
        image, bboxes, labels, difficult = self.dataset[index]
        image = torchvision.transforms.ToTensor()(image).to(config.device)#<-------------
        target = {}
        boxes = torch.tensor(data=bboxes,dtype=torch.float32).to(self.device)
        target["boxes"] = boxes if len(boxes.shape) > 1 else boxes.unsqueeze(dim=0)
        target["labels"] = torch.tensor(data=labels,dtype=torch.int64).to(self.device)
        return self.transform(image, target) if self.transform else (image, target)
    
    def __len__(self):
        return len(self.dataset) // self.fraction
    
    def collate_fn(self, data):
        return zip(*data)
