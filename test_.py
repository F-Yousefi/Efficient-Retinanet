from model.lightning_trainer import get_efficient_retinanet_trainer
from data.voc2007_dataset import generate_voc_dataloader


num_classes = 21
epochs = 40
batch_size = 4
backbone = "efficientnet-b4"


train_dataloader, val_dataloader = generate_voc_dataloader(
    batch_size=batch_size,
    num_workers=3,
    persistent_workers=True,
)

trainer, model = get_efficient_retinanet_trainer(
    backbone=backbone,
    epochs=epochs,
    batch_size=batch_size,
    trainable_backbone_layers=0,
    num_classes=num_classes,
    dir_to_save_model="ER_max_mAP",
)
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)
