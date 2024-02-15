import torch
import torchvision
from typing import Any
from torch.utils import data
from utils.config import config


class Dataset(data.Dataset):
    """
    class Dataset inhertited from torch.data.Dataset
        In this module, you should be careful about
        the collate_fn, because it is to manage how
        a batch should be merged together. The original
        one raises many errors.
    """

    def __init__(self, dataset, transform=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = "cpu"
        self.transform = transform

    def __getitem__(self, index) -> Any:
        image, bboxes, labels, difficult = self.dataset[index]
        image = torchvision.transforms.ToTensor()(image).to(self.device)
        target = {}
        boxes = torch.tensor(data=bboxes, dtype=torch.float32).to(self.device)
        target["boxes"] = boxes if len(boxes.shape) > 1 else boxes.unsqueeze(dim=0)
        target["labels"] = torch.tensor(data=labels, dtype=torch.int64).to(self.device)
        return self.transform(image, target) if self.transform else (image, target)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, data):
        return data
