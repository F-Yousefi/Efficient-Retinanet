import numpy as np
from torchvision.datasets import VOCDetection
from data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
import torch
import random

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




def transform(image, target):
    if random.randint(0, 1):
        image_size = image.shape[-2:]
        boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=image_size)
        composed_transforms = transforms.Compose([
                                    transforms.RandomRotation(60),
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



def generate_voc_dataloader(batch_size = 2, num_workers = 2, fraction =1):
    
    train_voc2007 = VOCDataset("../dataset/train_voc2007", "2007", "trainval")
    train_split = Dataset(train_voc2007, transform=transform, fraction= fraction)
    train_dataloader = DataLoader(dataset = train_split,
                                  batch_size= batch_size, 
                                  shuffle= False, 
                                  collate_fn= train_split.collate_fn, 
                                  num_workers=num_workers)
    val_voc2007 = VOCDataset("../dataset/test_voc2007", "2007", "test")
    val_split = Dataset(val_voc2007, fraction= fraction)
    val_dataloader = DataLoader(dataset= val_split,
                                batch_size= batch_size, 
                                shuffle= False,
                                collate_fn= val_split.collate_fn, 
                                num_workers=num_workers)
    return train_dataloader, val_dataloader