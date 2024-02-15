import os
from model.lightning_trainer import get_efficient_retinanet_trainer
from data.voc2007_dataset import generate_voc_dataloader
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    ResNet50_Weights,
)
from data.play_card_dataset import classes, generate_play_card_dataset

num_classes = len(classes)
epochs = 60
batch_size = 4
backbone = "resnet"

train_dataloader, val_dataloader = generate_play_card_dataset(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
)

# try:
for backbone in ["efficientnet-b4"]:

    if "efficientnet" in backbone:
        print(15 * "=", f"\n{backbone}-retinanet loaded.\n", 15 * "=")
        backbone_model = get_efficient_retinanet(
            num_classes=num_classes,
            efficientnet_model_name=backbone,
            trainable_backbone_layers=0,
            score_thresh=0.4,
        )
    elif "resnet" in backbone:
        print(15 * "=", f"\n{backbone}-retinanet loaded.\n", 15 * "=")
        backbone_model = retinanet_resnet50_fpn(
            weights=None,
            progress=True,
            num_classes=num_classes,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            trainable_backbone_layers=0,
            score_thresh=0.4,
        )

    trainer, model = get_efficient_retinanet_trainer(
        backbone=backbone_model,
        epochs=epochs,
        batch_size=batch_size,
        dir_to_save_model=f"ER_max_mAP_{backbone}",
        load="pre_trained_models/ER_max_mAP_efficientnet-b4.ckpt",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
# except:

# os.system("shutdown /s /t 300")
print("system will shutdown soon ...")


# os.system("shutdown /s /t 300")
print("system will shutdown soon ...")
