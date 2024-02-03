import torch
from data.dataset import generate_voc_dataloader
import torch
from tqdm import tqdm
from utils.config import config
from utils.gpu_check import gpu_temperature
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, ResNet50_Weights
try:
      train_dataloader, val_dataloader = generate_voc_dataloader(batch_size=4)
      model = retinanet_resnet50_fpn(    
          weights= None,
          progress=True,
          num_classes = 21,
          weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
          trainable_backbone_layers = 0).to(config.device)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
      _ = model.train()

      epochs = 20
      best_val_loss = [1000]
      patience = 10
      attempt = 0
      for epoch in range(1, epochs + 1):
          with tqdm(train_dataloader, unit="batch" ) as tepoch:
              loss_records = []
              for images, targets in tepoch:
                tepoch.set_description(f"Train      ep{epoch}")
                losses = model(images, targets)
                reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
                loss = (reg_loss + cls_loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_records.append(loss.item())
                tepoch.set_postfix(rloss=reg_loss.item(),
                                    sloss=cls_loss.item(),
                                    mean_loss = sum(loss_records) / len(loss_records))
                gpu_temperature(80,55)
                del loss, reg_loss, cls_loss, losses
                del images, targets
                torch.cuda.empty_cache()
                
          loss_records = []
          with tqdm(val_dataloader, unit="batch",) as tepoch:
              for images, targets in tepoch:
                tepoch.set_description(f"Validation ep{epoch}")
                losses = model(images, targets)
                reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
                loss = (reg_loss + cls_loss)
                loss_records.append(loss.item())
                tepoch.set_postfix(vrloss=reg_loss.item(),
                                    vsloss=cls_loss.item(),
                                    mean_vloss = sum(loss_records) / len(loss_records))
                gpu_temperature(80,55)
          if epoch == 5 or epoch == 10: 
            best_val_loss.append(sum(loss_records) / len(loss_records))
            torch.save(model.state_dict(), f"resnet-50-retinanet-{epoch}.pt")
            print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
            
          elif min(best_val_loss) > sum(loss_records) / len(loss_records):
            best_val_loss.append(sum(loss_records) / len(loss_records))
            torch.save(model.state_dict(), "resnet-50-retinanet.pt")
            print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
            attempt == 0
          else: 
            print("\n")
            attempt += 1

          if attempt >= patience:
            break

except:
      import os
      os.system("shutdown -a")
    
#===============================| RETINA-NET(efficientnet b4) |==========================================
import torch
from data.dataset import generate_voc_dataloader
import torch
from tqdm import tqdm
from utils.config import config
from utils.gpu_check import gpu_temperature
from model.efficient_retinanet import get_efficient_retinanet
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, ResNet50_Weights
name = "efficientnet-b2"
try:
      train_dataloader, val_dataloader = generate_voc_dataloader(batch_size=4)
      model = get_efficient_retinanet(21,name, trainable_backbone_layers=0).to(config.device)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
      _ = model.train()

      epochs = 20
      best_val_loss = [1000]
      patience = 10
      attempt = 0
      for epoch in range(1, epochs + 1):
          with tqdm(train_dataloader, unit="batch" ) as tepoch:
              loss_records = []
              for images, targets in tepoch:
                tepoch.set_description(f"Train      ep{epoch}")
                losses = model(images, targets)
                reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
                loss = (reg_loss + cls_loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_records.append(loss.item())
                tepoch.set_postfix(rloss=reg_loss.item(),
                                    sloss=cls_loss.item(),
                                    mean_loss = sum(loss_records) / len(loss_records))
                gpu_temperature(80,55)
                del loss, reg_loss, cls_loss, losses
                del images, targets
                torch.cuda.empty_cache()
                
          loss_records = []
          with tqdm(val_dataloader, unit="batch",) as tepoch:
              for images, targets in tepoch:
                tepoch.set_description(f"Validation ep{epoch}")
                losses = model(images, targets)
                reg_loss, cls_loss = losses["bbox_regression"] , losses["classification"]
                loss = (reg_loss + cls_loss)
                loss_records.append(loss.item())
                tepoch.set_postfix(vrloss=reg_loss.item(),
                                    vsloss=cls_loss.item(),
                                    mean_vloss = sum(loss_records) / len(loss_records))
                gpu_temperature(80,55)
          if epoch == 5 or epoch == 10: 
            best_val_loss.append(sum(loss_records) / len(loss_records))
            torch.save(model.state_dict(), f"{name}-retinanet-{epoch}.pt")
            print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
            
          elif min(best_val_loss) > sum(loss_records) / len(loss_records):
            best_val_loss.append(sum(loss_records) / len(loss_records))
            torch.save(model.state_dict(), f"{name}-retinanet.pt")
            print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
            attempt == 0
          else: 
            print("\n")
            attempt += 1

          if attempt >= patience:
            break

except:
      import os
      os.system("shutdown /s /t 60")

import os
os.system("shutdown -a")