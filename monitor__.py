import torchvision
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pylab as plt
from model.lightning_trainer import get_efficient_retinanet_trainer
from data.voc2007_dataset import generate_voc_dataloader
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    ResNet50_Weights,
)
from data.play_card_dataset import classes, generate_play_card_dataset

num_classes = len(classes)


_, test_dataloader = generate_play_card_dataset(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
)

backbone_model = get_efficient_retinanet(
    num_classes=num_classes,
    efficientnet_model_name="efficientnet-b4",
    trainable_backbone_layers=0,
    score_thresh=0.5,
    nms_thresh=0.01,
)

# backbone_model = retinanet_resnet50_fpn(
#     weights=None,
#     progress=True,
#     num_classes=num_classes,
#     weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
#     trainable_backbone_layers=0,
#     score_thresh=0.5,
# )

trainer, model = get_efficient_retinanet_trainer(
    backbone=backbone_model,
    epochs=None,
    batch_size=4,
    dir_to_save_model="ER_max_mAP_",
    load="ER_max_mAP_efficientnet-b4.ckpt",
)
detections = trainer.validate(model=model, dataloaders=test_dataloader)
print(len(detections))


detections = trainer.predict(model=model, dataloaders=test_dataloader)
print(len(detections))
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

# As you can see yourself, there are only eight features in our dataset that # have skewness under 1. To aquire a more accurate predict, we should reduce # the skewness of each feature as much as possible.
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8})
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(8, 16))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    ax.imshow(images_with_boxes[idx])
    plt.tight_layout()
plt.show()
