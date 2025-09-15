import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
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

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))
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
# PHASE 2: FULL FINE-TUNING
# ---------------------------
model_ft = get_model(NUM_CLASSES)
model_ft.load_state_dict(torch.load("/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/best_LPFT_model_epoch_19.pth"))
model_ft.to(device)

# Unfreeze backbone
for param in model_ft.backbone.parameters():
    param.requires_grad = True

# Differential learning rates
backbone_params = []
head_params = []

for name, param in model_ft.named_parameters():
    if 'backbone' in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer_ft = optim.SGD([
    {'params': backbone_params, 'lr': 0.001},
    {'params': head_params, 'lr': 0.005}
], momentum=0.9, weight_decay=0.0005, nesterov=True)

lr_scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

writer_ft = SummaryWriter("runs/retinanet_acls_ft")

num_epochs_ft = 200
# best_val_loss_ft = float('inf')
mAP_best = 0.499
with open("Log_file_LPFT_ACLS.txt", 'w') as lpft_log:
    for epoch in range(num_epochs_ft):
        print(f"FT epoch {epoch+1}")
        lpft_log.write(f"\nFT epoch {epoch+1}")
        lpft_log.flush()
        model_ft.train()
        train_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Training:\tEpoch {epoch+1}/{num_epochs_ft}")

        for images, targets in pbar:
            optimizer_ft.zero_grad()
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model_ft(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer_ft.step()

            train_loss += losses.item()

        lr_scheduler_ft.step()
        avg_train_loss = train_loss / len(train_loader)


        writer_ft.add_scalar("Loss/Train", avg_train_loss, epoch)

        model_ft.eval()
        output = []
        with torch.no_grad():
            metric.reset()
            pbar_val = tqdm.tqdm(val_loader, desc=f"Validation:\tEpoch {epoch+1}/{num_epochs_ft}")
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                output = model_ft(images)
                for out in output:
                        out['boxes'] = out['boxes'].detach().cpu()
                        out['scores'] = out['scores'].detach().cpu()
                        out['labels'] = out['labels'].detach().cpu()
                for t in targets:
                        t['boxes'] = t['boxes'].detach().cpu()
                        t['labels'] = t['labels'].detach().cpu()
                metric.update(output, targets)
            
            results = metric.compute()
            writer_ft.add_scalar("map50", results["map_50"], epoch)

            if results["map_50"] > mAP_best:
                mAP_best = results["map_50"]
                torch.save(model_ft.state_dict(), f"/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/best_LPFT_model_epoch_{epoch+1}.pth")
                print(f"✅ Saved new best model at epoch {epoch+1}")
                lpft_log.write(f"\n✅ Saved new best model at epoch {epoch+1}")
                lpft_log.flush()

        print(f"[Full Fine-tuning] Epoch {epoch+1}/{num_epochs_ft} | Train Loss: {avg_train_loss:.4f} = {train_loss}/{len(train_loader)} | mAP50 = {results['map_50']}")
        lpft_log.write(f"\n[Full Fine-tuning] Epoch {epoch+1}/{num_epochs_ft} | Train Loss: {avg_train_loss:.4f} = {train_loss}/{len(train_loader)} | mAP50 = {results['map_50']}")
        lpft_log.flush()

    # Save fine-tuned model
    torch.save(model_ft.state_dict(), "/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/retinanet_LPFT.pth")
    print("Full fine-tuned model saved.")
    writer_ft.close()
    lpft_log.write("Full fine-tuned model saved.")
