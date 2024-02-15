import torch
from typing import Any
import pytorch_lightning as L
from utils.config import config
from prettytable import PrettyTable
from utils.gpu_check import gpu_temperature
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler,
)


class EfficientRetinanetTrainer(L.LightningModule):
    def __init__(self, model, batch_size, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.model = model.to(config.device)
        self.mean_average_precision = MeanAveragePrecision()

    def training_step(self, batch) -> STEP_OUTPUT:
        images, targets = self.to_device(batch)
        self.model.train()
        losses = self.model(images, targets)
        reg_loss, cls_loss = losses["bbox_regression"], losses["classification"]
        loss = reg_loss + cls_loss
        self.log_dict(
            {"tloss": loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        gpu_temperature()
        return loss

    def on_train_epoch_end(self) -> None:
        print(end="\n")
        return super().on_train_epoch_end()

    def validation_step(self, batch) -> STEP_OUTPUT:
        images, targets = self.to_device(batch=batch)
        detections = [
            {key: value.detach().cpu() for key, value in image.items()}
            for image in self.model(images)
        ]
        targets = [
            {key: value.detach().cpu() for key, value in image.items()}
            for image in targets
        ]
        mAP = self.mean_average_precision(detections, targets)
        self.log(
            "val_mAP",
            mAP["map"],
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        del detections
        gpu_temperature()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        myTable = PrettyTable(["index", "value"])
        mAP = self.mean_average_precision.compute()
        _ = [
            myTable.add_row([key, f"{values.item()* 100:.1f}%"])
            for key, values in mAP.items()
            if values.nelement() == 1
        ]
        print(myTable)
        self.mean_average_precision.reset()

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        images, targets = self.to_device(batch=batch)
        detections = [
            {key: value.detach().cpu() for key, value in image.items()}
            for image in self.model(images)
        ]
        images = [image.detach().cpu() for image in images]
        return images, detections

    def to_device(self, batch):
        images = []
        targets = []
        for image, target in batch:
            image = image.to(config.device)
            target["boxes"] = target["boxes"].to(config.device)
            target["labels"] = target["labels"].to(config.device)
            images.append(image)
            targets.append(target)
        return images, targets

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=1e-05, weight_decay=1e-04)
