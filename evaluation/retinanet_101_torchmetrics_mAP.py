import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2


# Load config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/home/anjali/Documents/Anjali/RetinaNet_torch/output_retinanet101_LPFT_new/model_final.pth"               # ------------------------  comment if using pretrained model

# For finding mAP of the pretrained retinanet101 ---- Comment if using custom model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")


cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cuda" 

predictor = DefaultPredictor(cfg)

metric = MeanAveragePrecision(iou_type="bbox")


register_coco_instances("visdrone_test", {}, "/home/anjali/Documents/Anjali/openDET2/datasets/VisDrone_dataset/annotations/test_coco.json", "/home/anjali/Documents/Anjali/openDET2/datasets/VisDrone_dataset/test_human")

dataset_dicts = DatasetCatalog.get("visdrone_test")
metadata = MetadataCatalog.get("visdrone_test")

# Loop through dataset and accumulate predictions
for sample in dataset_dicts:
    img_path = sample["file_name"]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = predictor(img)
    
    instances = output["instances"].to("cpu")

    pred = {
        "boxes": instances.pred_boxes.tensor,
        "scores": instances.scores,
        "labels": instances.pred_classes
    }

    gt = {
        "boxes": torch.tensor([anno["bbox"] for anno in sample["annotations"]], dtype=torch.float),
        "labels": torch.tensor([anno["category_id"] for anno in sample["annotations"]])
    }

    # COCO bbox format is [x, y, width, height]; convert to [x1, y1, x2, y2]
    gt["boxes"][:, 2] += gt["boxes"][:, 0]
    gt["boxes"][:, 3] += gt["boxes"][:, 1]

    metric.update([pred], [gt])


final_results = metric.compute()
print(final_results)