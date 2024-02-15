"""This module let the main model start trainig."""

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as L
from data.play_card_dataset import generate_play_card_dataset, classes
from utils.config import config
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    ResNet50_Weights,
)
from model.lightning_trainer import EfficientRetinanetTrainer

__all__ = ["lunch"]

train_dataloader, val_dataloader = generate_play_card_dataset(
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    persistent_workers=True if config.num_workers > 0 else False,
)

if "resnet" in config.backbone_model_name:
    backbone_model = retinanet_resnet50_fpn(
        weights=None,
        progress=True,
        num_classes=len(classes),
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=0,
        score_thresh=0.5,
    )

elif "efficientnet" in config.backbone_model_name:
    backbone_model = get_efficient_retinanet(
        num_classes=len(classes),
        efficientnet_model_name=config.backbone_model_name,
        trainable_backbone_layers=0,
        score_thresh=0.5,
    )

else:
    print("The backbone model name is not defined.[resnet/efficientnet-b(0-7)]")


if config.dir_to_pretrained_model:
    model = EfficientRetinanetTrainer.load_from_checkpoint(
        checkpoint_path=config.dir_to_pretrained_model,
        model=backbone_model,
        batch_size=config.batch_size,
    )
else:
    model = EfficientRetinanetTrainer(
        model=backbone_model, batch_size=config.batch_size
    )


checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    filename=f"{config.backbone_model_name}_max_mAP",
    mode="max",
    dirpath=config.dir_to_save_model,
)

logger = CSVLogger(config.dir_to_log, name="retinanet_log.csv")

trainer = L.Trainer(
    accelerator=config.device,
    max_epochs=config.epochs,
    enable_model_summary=True,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback],
    logger=logger,
)


def lunch():
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
