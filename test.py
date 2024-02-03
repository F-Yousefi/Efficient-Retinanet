import torchvision
from PIL import Image
from data.dataset import CharlotteDataset
import torch
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.utils import draw_bounding_boxes
import matplotlib.pylab as plt
import numpy as np
from model.efficient_retinanet import retinanet_efficient_b0_fpn
from utils.config import config
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.datasets import VOCDetection
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, ResNet50_Weights

classes_list = np.array(CharlotteDataset().classes)
model = get_efficient_retinanet(21,"efficientnet-b4", trainable_backbone_layers=0,score_thresh=0.4 ).to(config.device)
# model = retinanet_resnet50_fpn(    
#     weights= None,
#     progress=True,
#     num_classes = 21,
#     weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
#     trainable_backbone_layers = 0, 
#     score_thresh=0.5).to(config.device)
model.eval()
# model.load_state_dict(torch.load("resnet-50-retinanet.pt", map_location=torch.device(config.device)))
model.load_state_dict(torch.load("efficientnet-b4-retinanet.pt", map_location=torch.device(config.device)))
while True:
    inp = input("Enter the name of the image:")
    image = Image.open(f"Screenshot ({inp}).png").convert("RGB") #503,502,462,452,447
    image = torchvision.transforms.ToTensor()(image).cuda()
    detections = model([image])
    res_img = draw_bounding_boxes(image=(image * 255).to(torch.uint8), 
                                boxes=detections[0]["boxes"],
                                width=5)
    print(detections[0]["labels"])
    plt.imshow(res_img.permute(1, 2, 0))
    plt.show()

