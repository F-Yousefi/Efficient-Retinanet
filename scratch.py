from typing import Any

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
from torch import Tensor
from data.charlotte_dataset import generate_charlotte_dataloader
from utils.gpu_check import gpu_temperature

from model.efficient_retinanet import get_efficient_retinanet

from prettytable import PrettyTable 
import pytorch_lightning as L
import torch
from typing import Any, List, Optional
from utils.config import config
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class CharlotteDetector(L.LightningModule):
    def __init__(self,batch, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch
        num_classes = 7
        model_name = "efficientnet-b4"
        trainable_backbone_layers = 0
        self.model = get_efficient_retinanet(num_classes=num_classes, 
                                            efficientnet_model_name=model_name, 
                                            trainable_backbone_layers=trainable_backbone_layers)
        # self.train_split_dataloader, self.val_split_dataloader, test = generate_charlotte_dataloader(
        #                                                 batch_size= self.batch_size, 
        #                                                 num_workers=4, 
        #                                                 train_frac=0.90, 
        #                                                 val_frac=0.099, 
        #                                                 test_frac=0.001)

        self.mean_average_precision  = MeanAveragePrecision()

    def training_step(self, batch) -> STEP_OUTPUT:
        images, targets = self.to_cuda(batch)
        self.model.train()
        losses = self.model(images, targets)
        reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
        loss = (reg_loss + cls_loss)
        self.log_dict({"train loss":loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        gpu_temperature(78, 55)
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_train_epoch_end(self) -> None:
        print(end='\n')
        return super().on_train_epoch_end()
    

    def validation_step(self, batch) -> STEP_OUTPUT:
        # pass
        images, targets = self.to_cuda(batch)
        self.model.eval()
        detections = self.model(images)
        # mAP = self.mean_average_precision(detections, targets)
        # self.log_dict({"validation mAP":mAP["map"]}, prog_bar=True, on_step=True, batch_size=self.batch_size)

    # def on_validation_epoch_end(self) -> None:
    #     myTable = PrettyTable(["index", "value"]) 
    #     mAP = self.mean_average_precision.compute()
    #     _ = [myTable.add_row([key, values]) for key, values in mAP.items()] 
    #     print(myTable)
    #     self.mean_average_precision.reset()


    

    def to_cuda(self, batch):
        images = []
        targets = []
        for (image, target) in batch:
            image = image.to(config.device)
            target["boxes"] = target["boxes"].to(config.device)
            target["labels"] = target["labels"].to(config.device)
            images.append(image)
            targets.append(target)
        return images, targets
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=1e-05, weight_decay=1e-04)

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return self.train_split_dataloader
    

batch_size = 2
train_split_dataloader, val_split_dataloader, test = generate_charlotte_dataloader(
                                                batch_size= batch_size, 
                                                num_workers=4, 
                                                train_frac=0.90, 
                                                val_frac=0.099, 
                                                test_frac=0.001)

model = CharlotteDetector(batch=batch_size)
trainer = L.Trainer(max_epochs = 10 , enable_model_summary=True, enable_progress_bar=True )
trainer.fit(model=model, train_dataloaders=train_split_dataloader, val_dataloaders=val_split_dataloader) 





# def to_cuda(batch):
#     images = []
#     targets = []
#     for (image, target) in batch:
#         image = image.to(config.device)
#         target["boxes"] = target["boxes"].to(config.device)
#         target["labels"] = target["labels"].to(config.device)
#         images.append(image)
#         targets.append(target)
#     return images, targets

# batch_size = 2
# num_classes = 7
# model_name = "efficientnet-b4"
# trainable_backbone_layers = 0
# model = get_efficient_retinanet(num_classes=num_classes, 
#                                     efficientnet_model_name=model_name, 
#                                     trainable_backbone_layers=trainable_backbone_layers).to("cuda")






# model.train()
# images, targets = to_cuda(next(iter(train_split_dataloader)))
# loss = model(images, targets)
# model.eval()
# model.training
# images, targets = to_cuda(next(iter(val_split_dataloader)))
# detection = model(images)