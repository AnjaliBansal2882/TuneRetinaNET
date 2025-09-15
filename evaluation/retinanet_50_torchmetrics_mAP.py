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
            labels.append(obj["category_id"]+1)

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
    

val_img_folder = "/home/botlab/Documents/Anjali/openDET2/datasets/VisDrone_dataset/test_human"
val_ann_file = "/home/botlab/Documents/Anjali/openDET2/datasets/VisDrone_dataset/annotations//.json"

val_dataset = CocoRetinaNetDataset(val_img_folder, val_ann_file, transforms=T.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))


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

model_ft = get_model(NUM_CLASSES)
model_ft.load_state_dict(torch.load("/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/best_ACLS_model_LPFT_epoch19.pth"))
model_ft.to(device)


model_ft.eval()
output = []
with torch.no_grad():
    pbar_val = tqdm.tqdm(val_loader, desc=f"Validation:")
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
print(results)
