from utils.gpu_check import gpu_temperature
from prettytable import PrettyTable 
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

#------model------
from model.efficient_retinanet import get_efficient_retinanet

num_classes = 7
model_name = "efficientnet-b4"
trainable_backbone_layers = 0
model = get_efficient_retinanet(num_classes=num_classes, 
                                    efficientnet_model_name=model_name, 
                                    trainable_backbone_layers=trainable_backbone_layers,
                                    score_thresh=0.4).to("cuda")

#-----dataset-------
from data.charlotte_dataset import generate_charlotte_dataloader, to_cuda
batch_size = 2
train_dataloader , val_dataloader = generate_charlotte_dataloader(train_frac=0.9, 
                                                                   val_frac=0.1, 
                                                                   num_workers = 8, 
                                                                   batch_size = batch_size)
# images, targets = to_cuda(next(iter(val_dataloader)))
#----------test--------------
# import time
# _ = model.eval()
# detections_list = []
# targets_list = []
# for i in range(20):
#     if i < 20 :
#         images, targets = to_cuda(next(iter(val_dataloader)))
#         detections = model(images)
#         detections_list.extend([{key: value.detach().cpu() for key, value in image.items()}  for image in detections ])
#         targets_list.extend([{key: value.detach().cpu() for key, value in image.items()}  for image in targets ])
#         time.sleep(2)
#         print(i)
#     else:
#         del model
#         time.sleep(20)


# del targets_list
# torch.cuda.empty_cache()
#     # mAP = mean_average_precision(detections, targets)
#     # tepoch.set_description(f"Validation ep{epoch}")
#     # tepoch.set_postfix(validation_mAP = mAP["map"])
#     # myTable = PrettyTable(["index", "value"]) 
#     # _ = [myTable.add_row([key, values]) for key, values in mAP.items()] 
#     # print(myTable)
#     # torch.cuda.empty_cache()
# import torch 
# v = torch.rand((100,300,300,200), device="cuda")
# del v
#------training----------
from tqdm import tqdm
import torch 
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from prettytable import PrettyTable 


optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=1e-04)
epochs = 1
best_val_loss = [1000]
patience = 10
attempt = 0
mean_average_precision = MeanAveragePrecision()

for epoch in range(1, epochs + 1):
    with tqdm(train_dataloader, unit="batch" ) as tepoch:
        _ = model.train()
        loss_records = []
        for batch in tepoch:
            images, targets = to_cuda(batch=batch)
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

        
    loss_records = []
    with tqdm(val_dataloader, unit="batch",) as tepoch:
        _ = model.eval()
        for batch in tepoch:
            images, targets = to_cuda(batch=batch)
            detections = [{key: value.detach().cpu() for key, value in image.items()}  for image in model(images)]
            targets = [{key: value.detach().cpu() for key, value in image.items()}  for image in targets]
            mAP = mean_average_precision(detections, targets)
            tepoch.set_description(f"Validation ep{epoch}")
            tepoch.set_postfix(validation_mAP = mAP["map"])
            gpu_temperature(80,55)
            del detections
            torch.cuda.empty_cache()

        myTable = PrettyTable(["index", "value"]) 
        mAP = mean_average_precision.compute()
        _ = [myTable.add_row([key, values]) for key, values in mAP.items()] 
        print(myTable)
        mean_average_precision.reset()

    if epoch == 5 or epoch == 10: 
        best_val_loss.append(sum(loss_records) / len(loss_records))
        # torch.save(model.state_dict(), f"resnet-50-retinanet-{epoch}.pt")
        print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
    
    elif min(best_val_loss) > sum(loss_records) / len(loss_records):
        best_val_loss.append(sum(loss_records) / len(loss_records))
        # torch.save(model.state_dict(), "resnet-50-retinanet.pt")
        print(f"saved! - best validation loss: {min(best_val_loss):.5f} \n")
        attempt == 0
    else: 
        print("\n")
        attempt += 1

    if attempt >= patience:
        break


    #     images, targets = self.to_cuda(batch)
    #     self.model.eval()
    #     detections = self.model(images)
    #     # mAP = self.mean_average_precision(detections, targets)
    #     # self.log_dict({"validation mAP":mAP["map"]}, prog_bar=True, on_step=True, batch_size=self.batch_size)

    # # def on_validation_epoch_end(self) -> None:
    # #     myTable = PrettyTable(["index", "value"]) 
    # #     mAP = self.mean_average_precision.compute()
    # #     _ = [myTable.add_row([key, values]) for key, values in mAP.items()] 
    # #     print(myTable)
    # #     self.mean_average_precision.reset()