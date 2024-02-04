import os
from typing import Any, Tuple
from utils.config import config
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as ET_Element
import collections
from PIL import Image
import numpy as np
from torch.utils import data
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import random

from torchvision.datasets import VOCDetection



class CharlotteDataset():
    def __init__(self):
        self.path = Path(config.dir_to_dataset)
        self.annotation_trees = []
        with open(self.path / "classes.txt", "r") as cls_file:
            self.classes = cls_file.read().split("\n")
            self.classes = ["__background__"] + self.classes

        for filename in os.listdir(self.path):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(self.path, filename)
            if '<object>' not in open(fullname).read(): continue
            tree = ET.parse(fullname)
            self.annotation_trees.append(tree)
        self.len = len(self.annotation_trees)
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        voc_dict = self.parse_voc_xml(self.annotation_trees[idx].getroot())
        ann = voc_dict["annotation"]
        image = Image.open(os.path.join(self.path, ann["path"].split("\\")[-1])).convert("RGB")
        bboxes = []
        labels = []
        difficult = []
        for obj in ann["object"]:
          xmin, ymin, xmax, ymax = [int(i) for i in obj["bndbox"].values()]

          if obj["name"] in self.classes:
              cat = self.classes.index(obj["name"])
              labels.append(cat)
          else:
            m = obj["name"]
            print(m)
            raise f"the class {m} is not defined."
          bboxes.append(([xmin, ymin, xmax, ymax]) )
          difficult.append(obj["difficult"])
        return image, np.array(bboxes), np.array(labels), np.array(difficult)
    
    @staticmethod
    def parse_voc_xml(node: ET_Element):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(CharlotteDataset.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict



class VOCDataset(VOCDetection):
    classes = ["__background__","person","bird", "cat", "cow", "dog", "horse", "sheep",\
               "aeroplane", "bicycle", "boat", "bus", "car", "motorbike",\
               "train","bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",]
    def __getitem__(self, idx):
        image, target =  super().__getitem__(idx)
        ann = target["annotation"]
        bboxes = []
        labels = []
        difficult = []
        for obj in ann["object"]:
          xmin, ymin, xmax, ymax = [int(i) for i in obj["bndbox"].values()]

          if obj["name"] in self.classes:
              cat = self.classes.index(obj["name"])
              labels.append(cat)
          else:
            m = obj["name"]
            print(m)
            raise f"the class {m} is not defined."
          bboxes.append(([xmin, ymin, xmax, ymax]) )
          difficult.append(obj["difficult"])
        return image, np.array(bboxes), np.array(labels), np.array(difficult)
    

 
class Dataset(data.Dataset):
    def __init__(self, dataset, transform = None, fraction = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = config.device
        self.fraction = fraction
        if transform: self.active_transform = True
        else : self.active_transform = False

    def __getitem__(self, index) -> Any:
        image, bboxes, labels, difficult = self.dataset[index]
        image = torchvision.transforms.ToTensor()(image).to(config.device)#<-------------
        target = {}
        boxes = torch.tensor(data=bboxes,dtype=torch.float32).to(self.device)
        target["boxes"] = boxes if len(boxes.shape) > 1 else boxes.unsqueeze(dim=0)
        target["labels"] = torch.tensor(data=labels,dtype=torch.int64).to(self.device)
        return self.transform(image, target) if self.active_transform else (image, target)
    
    def __len__(self):
        return len(self.dataset) // self.fraction
    
    def transform(self, image, target):
        if random.randint(0, 1):
            image_size = image.shape[-2:]
            boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=image_size)
            composed_transforms = transforms.Compose([
                                        transforms.RandomRotation(10),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(image_size, scale=(0.9,1), antialias=True),
                                        transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.1,
                                                                saturation=0.3, 
                                                                hue=0)
                                    ])
            tr_image, tr_boxes = composed_transforms(image, boxes)
            not_zero_boxes = []
            for i in range(len(tr_boxes)):
                if abs(tr_boxes[i][0] - tr_boxes[i][2]) > 0 and abs(tr_boxes[i][1] - tr_boxes[i][3]) > 0:
                    not_zero_boxes.append(tr_boxes[i])
            
            if len(not_zero_boxes) > 0 :
                target["boxes"] = torch.stack(not_zero_boxes) 
            else: 
                return image, target
            
            return tr_image, target
            
        else:
            return image, target




def prepare_batch(data):
    images = [data[i][0] for i in range(len(data))]
    targets = [{"boxes": data[i][1]["boxes"],"labels": data[i][1]["labels"] } for i in range(len(data))]
    # print(targets[0]["boxes"].shape,targets[1]["boxes"].shape,targets[2]["boxes"].shape)

    return images, targets

def generate_charlotte_dataloader(batch_size = 2 ,train_frac =0.02,val_frac = 0.02, test_frac = 0.96):
    charlotte_dataset = CharlotteDataset()
    train_split, val_split , test_split = random_split(charlotte_dataset, (train_frac,val_frac, test_frac))
    train_split = Dataset(train_split, transform=True)
    val_split = Dataset(val_split)
    test_split = Dataset(test_split)
    train_dataloader = DataLoader(train_split,batch_size, True,collate_fn=prepare_batch)
    val_dataloader = DataLoader(val_split,batch_size, False,collate_fn=prepare_batch)
    test_dataloader = DataLoader(test_split,batch_size, False,collate_fn=prepare_batch)
    return train_dataloader, val_dataloader, test_dataloader


def generate_voc_dataloader(batch_size = 2):
    train_voc2007 = VOCDataset("../dataset/train_voc2007", "2007", "train")
    train_split = Dataset(train_voc2007, transform=True, fraction= 5)
    val_voc2007 = VOCDataset("../dataset/test_voc2007", "2007", "test")
    val_split = Dataset(train_voc2007, fraction= 5)
    train_dataloader = DataLoader(train_split,batch_size, False, collate_fn=prepare_batch, num_workers=2)
    val_dataloader = DataLoader(val_split,batch_size, False,collate_fn=prepare_batch,num_workers=2)
    return train_dataloader, val_dataloader




# def transform(self, image, target):
#         if random.randint(0, 1):
#             image_size = image.shape[-2:]
#             boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=image_size)
#             composed_transforms = transforms.Compose([
#                                         transforms.RandomRotation(60),
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.RandomResizedCrop(image_size, scale=(0.9,1), antialias=True),
#                                         transforms.ColorJitter(brightness=0.5,
#                                                                 contrast=0.1,
#                                                                 saturation=0.3, 
#                                                                 hue=0)
#                                     ])
#             tr_image, tr_boxes = composed_transforms(image, boxes)
#             not_zero_boxes = []
#             for i in range(len(tr_boxes)):
#                 if abs(tr_boxes[i][0] - tr_boxes[i][2]) > 0 and abs(tr_boxes[i][1] - tr_boxes[i][3]) > 0:
#                     not_zero_boxes.append(tr_boxes[i])
            
#             if len(not_zero_boxes) > 0 :
#                 target["boxes"] = torch.stack(not_zero_boxes) 
#             else: 
#                 return image, target
            
#             return tr_image, target
            
#         else:
#             return image, target