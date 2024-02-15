"""This module is specifically built for the play card dataset which has been uploaded along the project."""

import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict
from pathlib import Path
from utils.config import config
from data.dataset import Dataset
from torchvision import tv_tensors
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

classes = ["__background__", "ace", "jack", "king", "nine", "queen", "ten"]
__all__ = ["train_dataloader", "val_dataloader"]


class PlaycardDataset:
    """
    class PlayDataset:
        This class is specifically built for the play card dataset which
        has been uploaded along the project.
        This class can be a good example of how you can use your desired
        dataset.
        Whatever dataset that you are going to use, just be careful to
        return the same output as this class.
    """

    def __init__(self, dir_to_dataset, train: bool = True) -> None:
        mode = "train" if train == True else "test"
        self.dir_to_dataset = dir_to_dataset
        self.dataset_table = pd.read_csv(Path(dir_to_dataset) / f"{mode}_labels.csv")
        self.image_path_list = list(
            self.dataset_table.groupby("filename").groups.keys()
        )
        self.class_map = ["__background__"] + list(
            self.dataset_table.groupby("class").groups.keys()
        )

    def __getitem__(self, index: int):
        targets = self.dataset_table[
            self.dataset_table["filename"] == self.image_path_list[index]
        ]
        image = Image.open(
            Path(self.dir_to_dataset) / self.image_path_list[index]
        ).convert("RGB")
        boxes = []
        labels = []
        for idx, target in targets.iterrows():
            labels.append(self.class_map.index(target["class"]))
            boxes.append(list(target[target.keys()[-4:]]))

        return image, np.array(boxes), np.array(labels), None

    def __len__(self):
        return len(self.image_path_list)


def transform(image: Image, target: Dict):
    if random.randint(0, 1):
        image_size = image.shape[-2:]
        boxes = tv_tensors.BoundingBoxes(
            target["boxes"], format="XYXY", canvas_size=image_size
        )
        composed_transforms = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.9, 1), antialias=True
                ),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.1, saturation=0.3, hue=0
                ),
            ]
        )
        tr_image, tr_boxes = composed_transforms(image, boxes)
        not_zero_boxes = []
        for i in range(len(tr_boxes)):
            if (
                abs(tr_boxes[i][0] - tr_boxes[i][2]) > 0
                and abs(tr_boxes[i][1] - tr_boxes[i][3]) > 0
            ):
                not_zero_boxes.append(tr_boxes[i])

        if len(not_zero_boxes) > 0:
            target["boxes"] = torch.stack(not_zero_boxes)
        else:
            return image, target

        return tr_image, target

    else:
        return image, target


def generate_play_card_dataset(batch_size=2, num_workers=2, **kwargs):

    train_play_card = PlaycardDataset(
        Path(config.dir_to_dataset) / "play_cards", train=True
    )
    train_split = Dataset(train_play_card, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_split.collate_fn,
        num_workers=num_workers,
        **kwargs,
    )
    val_play_card = PlaycardDataset(
        Path(config.dir_to_dataset) / "play_cards", train=False
    )
    val_split = Dataset(val_play_card)
    val_dataloader = DataLoader(
        dataset=val_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_split.collate_fn,
        num_workers=num_workers,
        **kwargs,
    )
    return train_dataloader, val_dataloader
