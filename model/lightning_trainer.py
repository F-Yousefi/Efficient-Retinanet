import torch
from typing import Any, Optional
import pytorch_lightning as L
from utils.config import config
from prettytable import PrettyTable
from utils.gpu_check import gpu_temperature
from pytorch_lightning.callbacks import ModelCheckpoint
from model.efficient_retinanet import get_efficient_retinanet
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
            {"train loss": loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        gpu_temperature(config.high_temp, config.low_temp)
        return super().on_train_batch_end(outputs, batch, batch_idx)

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
            "validation_mAP",
            mAP["map"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        gpu_temperature(config.high_temp, config.low_temp)
        del detections
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

    def predict_step(self, images, *args: Any, **kwargs: Any) -> Any:
        images = images.to("cuda")
        detections = [
            {key: value.detach().cpu() for key, value in image.items()}
            for image in self.model(images)
        ]
        return detections

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


def get_efficient_retinanet_trainer(
    backbone,
    epochs,
    batch_size,
    num_classes,
    trainable_backbone_layers,
    dir_to_save_model,
    **kwargs: Any,
) -> (Optional[L.Trainer], Optional[L.LightningModule]):
    backbone = get_efficient_retinanet(
        num_classes=num_classes,
        efficientnet_model_name=backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
    model = EfficientRetinanetTrainer(model=backbone, batch_size=batch_size, **kwargs)
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_mAP", filename=dir_to_save_model, mode="max", dirpath="./"
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        enable_model_summary=True,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
    )
    return trainer, model
