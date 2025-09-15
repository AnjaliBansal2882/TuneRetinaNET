import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, retinanet_res
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn as nn
import math
import cv2 as cv
import numpy as np
from functools import partial

# Dataset
class CocoRetinaNetDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoRetinaNetDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoRetinaNetDataset, self).__getitem__(idx)

        # if self._transforms:
        #     img = self._transforms(img)
        # else:
        #     img = T.ToTensor()(img)

        boxes = []
        labels = []

        for obj in target:
            xmin, ymin, width, height = obj["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(obj["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        coco_target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            sample = self.transforms(image = img,
                                     bboxes = coco_target['boxes'],
                                     labels = labels)
            img = sample['image']
            coco_target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((coco_target['boxes']).numpy()).any() or coco_target['boxes'].shape == torch.Size([0]):
            coco_target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        img = T.ToTensor()(img)
        return img, coco_target

# Paths
train_img_folder = "/home/botlab/Documents/Anjali/openDET2/datasets/VisDrone_dataset/train"
train_ann_file = "/home/botlab/Documents/Anjali/train_coco.json"

val_img_folder = "/home/botlab/Documents/Anjali/openDET2/datasets/VisDrone_dataset/val"
val_ann_file = "/home/botlab/Documents/Anjali/val_coco.json"

# Dataset and loaders
train_dataset = CocoRetinaNetDataset(train_img_folder, train_ann_file, transforms=T.ToTensor())
val_dataset = CocoRetinaNetDataset(val_img_folder, val_ann_file, transforms=T.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))

# Model
CLASSES = ['__background__', 'person']
NUM_CLASSES = len(CLASSES)  # Modify if needed

def get_model(NUM_CLASSES):
    # model = retinanet_resnet50_fpn_v2(pretrained=True)
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    )

    in_features = model.backbone.out_channels  # The correct input channels for the detection head
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    print(model.head.classification_head.cls_logits.weight.shape)

    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, bias_value)
    
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric = MeanAveragePrecision()

# ---------------------------
# PHASE 1: LINEAR PROBING
# ---------------------------
model = get_model(NUM_CLASSES)


# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

model.backbone.eval()

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov = True)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# TensorBoard
# print("hello")
writer = SummaryWriter("runs/retinanet_acls_lp")

num_epochs_linear = 50
best_val_loss_lp = float('inf')
mAP_best = 0
with open("Log_file_LP_ACLS.txt", 'a+') as log:

    for epoch in range(num_epochs_linear):
        print(f"epoch {epoch+1} for LP")
        log.write(f"\nepoch {epoch+1} for LP")
        log.flush()
        model.train()
        train_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Training:\tEpoch {epoch+1}/{num_epochs_linear}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())               # The loss_dict conatins 2 types of losses:  classification:Tensor() and bbox_regression:Tensor()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        # model.train()
        # val_loss = 0.0
        # with torch.no_grad():
        #     pbar_val = tqdm.tqdm(val_loader, desc=f"Validation:\tEpoch {epoch+1}/{num_epochs_linear}")
        #     for images, targets in pbar_val:
        #         images = [img.to(device) for img in images]
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #         # Use the same loss computation during val
        #         loss_dict = model(images, targets)
        #         for t in targets:
        #             if (t["labels"] >= NUM_CLASSES).any():
        #                 print("🚨 Invalid label found:", t["labels"])
        #         losses = sum(loss for loss in loss_dict.values())

        #         val_loss += losses.item()

        # avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        # writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        
        # if avg_val_loss < best_val_loss_lp:
        #     best_val_loss_lp = avg_val_loss
        #     torch.save(model.state_dict(), f"/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints/best_model_epoch_{epoch+1}.pth")
        #     print(f"✅ Saved new best model at epoch {epoch+1}")

        model.eval()
        output = []
        with torch.no_grad():
            metric.reset()
            pbar_val = tqdm.tqdm(val_loader, desc=f"Validation:\tEpoch {epoch+1}/{num_epochs_linear}")
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                output = model(images)
                for out in output:
                        out['boxes'] = out['boxes'].detach().cpu()
                        out['scores'] = out['scores'].detach().cpu()
                        out['labels'] = out['labels'].detach().cpu()
                for t in targets:
                        t['boxes'] = t['boxes'].detach().cpu()
                        t['labels'] = t['labels'].detach().cpu()
                metric.update(output, targets)

            results = metric.compute()
            writer.add_scalar("map50", results["map_50"], epoch)
                

            if mAP_best < results["map_50"]:
                mAP_best = results["map_50"]
                torch.save(model.state_dict(), f"/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/best_model_epoch_{epoch+1}.pth")
                print(f"✅ Saved new best model at epoch {epoch+1}")
                log.write(f"\n✅ Saved new best model at epoch {epoch+1}")
                log.flush()

        # print(f"[Linear Probing] Epoch {epoch+1}/{num_epochs_linear} | Train Loss: {avg_train_loss:.4f} = {train_loss}/{len(train_loader)} | Val Loss: {avg_val_loss:.4f} = {val_loss}/{len(val_loader)} | mAP50: {results["map_50"]}")
        print(f"[Linear Probing] Epoch {epoch+1}/{num_epochs_linear} | Train Loss: {avg_train_loss:.4f} = {train_loss}/{len(train_loader)} | mAP50: {results["map_50"]}")
        log.write(f"\n[Linear Probing] Epoch {epoch+1}/{num_epochs_linear} | Train Loss: {avg_train_loss:.4f} = {train_loss}/{len(train_loader)} | mAP50: {results["map_50"]}")
        log.flush()

    # Save linear probing model
    torch.save(model.state_dict(), "/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/retinanet_ACLS_LP.pth")
    print("Linear probing model saved.")
    log.write("Last LP model saved")
    log.flush()
    writer.close()
