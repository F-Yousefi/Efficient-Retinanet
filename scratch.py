from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
from torch import Tensor
from data.charlotte_dataset import generate_charlotte_dataloader
from data.voc2007_dataset import generate_voc_dataloader
from utils.gpu_check import gpu_temperature

from model.efficient_retinanet import get_efficient_retinanet


import pytorch_lightning as L
import torch
from typing import Any, List, Optional

train_split_dataloader, val_split_dataloader, test = generate_charlotte_dataloader(
                                                        batch_size= 12, 
                                                        num_workers=0, 
                                                        train_frac=0.90, 
                                                        val_frac=0.099, 
                                                        test_frac=0.001)

images, targets = next(iter(train_split_dataloader))













class CharlotteDetector(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        num_classes = 7
        model_name = "efficientnet-b4"
        trainable_backbone_layers = 0
        self.model = get_efficient_retinanet(num_classes=num_classes, 
                                            efficientnet_model_name=model_name, 
                                            trainable_backbone_layers=trainable_backbone_layers)
        self.train_split_dataloader, self.val_split_dataloader, test = generate_charlotte_dataloader(
                                                        batch_size= 4, 
                                                        num_workers=10, 
                                                        train_frac=0.90, 
                                                        val_frac=0.099, 
                                                        test_frac=0.001)
    def training_step(self, batch) -> STEP_OUTPUT:
        images, targets = batch
        losses = self.model(images, targets)
        reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
        loss = (reg_loss + cls_loss)
        self.log_dict({"loss":loss, })
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        gpu_temperature(78, 55)
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=1e-05, weight_decay=1e-04)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_split_dataloader

model = CharlotteDetector()
trainer = L.Trainer(max_epochs = 1 , enable_model_summary=True, enable_progress_bar=True )
trainer.fit(model=model) 