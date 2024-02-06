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
                                            trainable_backbone_layers=trainable_backbone_layers,
                                            score_thresh=0.4).to("cuda")
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
        images, targets = self.to_cuda(batch=batch)
        detections = [{key: value.detach().cpu() for key, value in image.items()}  for image in self.model(images)]
        targets = [{key: value.detach().cpu() for key, value in image.items()}  for image in targets]
        mAP = self.mean_average_precision(detections, targets)
        self.log("validation_mAP", mAP["map"], prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        gpu_temperature(80,55)
        del detections
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        myTable = PrettyTable(["index", "value"]) 
        mAP = self.mean_average_precision.compute()
        _ = [myTable.add_row([key, f"{values.item():.4f}"]) for key, values in mAP.items() if values.nelement() == 1 ] 
        print(myTable)
        self.mean_average_precision.reset()
    
    def predict_step(self, images, *args: Any, **kwargs: Any) -> Any:
        images = images.to("cuda")
        detections = [{key: value.detach().cpu() for key, value in image.items()}  for image in self.model(images)]
        return detections
    

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

    

from data.charlotte_dataset import generate_charlotte_dataloader, to_cuda
batch_size = 4
train_dataloader , val_dataloader = generate_charlotte_dataloader(train_frac=0.9, 
                                                                   val_frac=0.1, 
                                                                   num_workers = 3, 
                                                                   batch_size = batch_size,
                                                                   persistent_workers=True)

from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(monitor='validation_mAP', filename="ER_best_mAP.ckpt", mode="max",dirpath="./")

model = CharlotteDetector(batch=batch_size)
trainer = L.Trainer(max_epochs = 20 , enable_model_summary=True, enable_progress_bar=True, callbacks=[checkpoint_callback] )
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader) 


model =CharlotteDetector.load_from_checkpoint("ER_best_mAP.ckpt", batch=batch_size)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model = CharlotteDetector(batch=batch_size)
trainer = L.Trainer(max_epochs = 20 , enable_model_summary=True, enable_progress_bar=True, callbacks=[checkpoint_callback] )
trainer.validate(model, ckpt_path="ER_best_mAP.ckpt", dataloaders=val_dataloader)