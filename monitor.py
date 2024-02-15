"""This module is to monitor the performance of Efficient-Retinanet."""

import warnings
import pytorch_lightning as L
import matplotlib.pylab as plt
from utils.config import config
from torchvision.utils import draw_bounding_boxes
from model.lightning_trainer import EfficientRetinanetTrainer
from model.efficient_retinanet import get_efficient_retinanet
from data.play_card_dataset import generate_play_card_dataset, classes
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    ResNet50_Weights,
)

warnings.filterwarnings("ignore")
import torch


_, test_dataloader = generate_play_card_dataset(
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
    pass

if config.dir_to_pretrained_model != None:
    print(f"model placed in '{config.dir_to_pretrained_model}' is loaded. ")
    model = EfficientRetinanetTrainer.load_from_checkpoint(
        checkpoint_path=config.dir_to_pretrained_model,
        model=backbone_model,
        batch_size=config.batch_size,
    )
else:
    model = EfficientRetinanetTrainer(
        model=backbone_model, batch_size=config.batch_size
    )

trainer = L.Trainer(
    accelerator=config.device,
    max_epochs=config.epochs,
    enable_model_summary=True,
    enable_progress_bar=True,
)


def lunch():

    detections = trainer.predict(model=model, dataloaders=test_dataloader)
    images_with_boxes = []
    for images, detection in detections:
        for idx in range(len(images)):
            res_img = draw_bounding_boxes(
                image=(images[idx] * 255).to(torch.uint8),
                boxes=detection[idx]["boxes"],
                labels=[classes[i] for i in detection[idx]["labels"]],
                width=5,
            )
            images_with_boxes.append(res_img.permute(1, 2, 0))

    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.imshow(images_with_boxes[idx])
        plt.tight_layout()
    plt.show()
