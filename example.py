"This module just shows the simplest way that you can use the model in your projects."

# ------Importing required libraries----
from prettytable import PrettyTable
import torch
from model.efficient_retinanet import get_efficient_retinanet

# -----------Loading dataset------------
device = "cuda"
batch_size = 4
num_classes = 7  # +[__background__]
images = [torch.rand((3, 400, 400), device=device) for i in range(batch_size)]
targets = [
    {
        "boxes": torch.tensor(
            [[1, 2, 40, 50], [4, 3, 60, 80]], device=device, dtype=torch.int32
        ),
        "labels": torch.tensor([1, num_classes - 1], device=device, dtype=torch.int32),
    }
    for i in range(batch_size)
]

# -----------Creating Model--------------
model = get_efficient_retinanet(
    num_classes=num_classes,
    efficientnet_model_name="efficientnet-b4",
    trainable_backbone_layers=0,
    score_thresh=0.5,
).to(device=device)


# ------------Training process------------
_ = model.train()
loss = model(images, targets)
print(f"\n * The loss of the dummy dataset is equal to:")
loss_table = PrettyTable(["Loss", "value"])
for key, value in loss.items():
    loss_table.add_row([key, f"{value.item():.4f}"])
print(loss_table)

# ------------Predicing process------------
_ = model.eval()
detections = model(images)
print(
    f"\n * The inference of the model is a dict instance with\n   the following keys:{detections[0].keys()}"
)
