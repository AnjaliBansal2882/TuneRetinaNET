# Script to calculate the FLOPs of any retinanet model using the fvcore lib

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
from fvcore.nn import FlopCountAnalysis

# Build and setup the model with the required head

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

model_ft = get_model(NUM_CLASSES)
model_ft.load_state_dict(torch.load("/home/botlab/Documents/Anjali/RetinaNet_torch/checkpoints_ACLS_mAP/best_ACLS_model_LPFT_epoch19.pth"))
model_ft.to(device)


model_ft.eval()

# Count FLOPs in a single forward pass on dummy input

input_tensor = torch.randn(1, 3, 800, 800).to(device)  # Adjust size if needed
flops = FlopCountAnalysis(model_ft, input_tensor)


forward_flops = flops.total()
training_flops = forward_flops * 3

print(f"Forward FLOPs: {forward_flops / 1e9:.2f} GFLOPs")
print(f"Training FLOPs (est): {training_flops / 1e9:.2f} GFLOPs")

