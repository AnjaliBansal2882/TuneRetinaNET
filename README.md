# TuneRetinaNET

This was done with the aim of getting much hihger mAP50 values on the Visdrone dataset in object detection than YOLO models, for accessing the future scope of performing Knowledge distillation for edge devices. The repo is a complete guide to using the retinanet as per the custom requirements.

Scripts for fine tuning RetinaNET with different backbones and custom heads on custom datasets. 

I have used 2 types of backbones. Namely:
- ResNet50 pretrained on ImageNet Dataset found in torchvision
- ResNet101 pretrained found in Detectron2

The head needs to be changed for the number of target classes and environment.

For comparing model efficiency with other models, we also compute the FLOPs using the **fvcore** lib of all retinanet models.

I also computed the mAP for these fine tunings. The scripts for the same can be found in repo/evaluation. The results for the 
| RetinaNET50 Model Cases|   LP  |  LPFT |
|------------------------|-------|-------|
| Fine tuned on VisDrone | 0.194 | 0.232 |
| Fine tuning + ACLS loss| 0.198 | 0.250 |

| RetinaNET101 Model Cases | LP    | LPFT  |
|--------------------------|-------|-------|
| Fne Tuned on VisDrone    | 0.197 | 0.223 |



