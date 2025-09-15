import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

# register visdrone dataset havig only 1 class
register_coco_instances("visdrone_train", {}, "/path/to/training/annotatations/in/json/train_coco.json", "/path/to/train/images")
register_coco_instances("visdrone_val", {}, "/path/to/validating/annotatations/in/json/val_coco.json", "/path/to/val/images")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x")

cfg.DATASETS.TRAIN = ("visdrone_train",)
cfg.DATASETS.TEST = ("visdrone_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001  # For linear probing
cfg.SOLVER.MAX_ITER = 6000
cfg.SOLVER.STEPS = []

cfg.MODEL.RETINANET.NUM_CLASSES = 1  
cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR = "./output_retinanet101/"
cfg.MODEL.BACKBONE.FREEZE_AT = 5 



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
