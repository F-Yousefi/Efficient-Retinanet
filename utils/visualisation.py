from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import cv2


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def draw_boxes(image, boxes, color=(0, 0, 255)):
    img = np.array(image)
    for b in boxes:
        img = cv2.rectangle(img, b[:2], b[2:], color=color, thickness=2)
    plt.imshow(img)
    plt.show()
