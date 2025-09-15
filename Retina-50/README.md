The above scripts:
## Performs Linear Probing/ Full fine tuning/ linear probing followed by full fine tuning of the Torchvision Retinanet model 
- it has resnet50 as its backbone
- also has FPN backbone along with it, trained on imagenet dataset
- on Visdrone Dataset for aerial images
- trained on only person class
- uses COCO style formatted dataset and hence, COCODetectionAPI by torchvision
- Performs validation after each epoch on the validation set
- Build an GUI output for easy user understanding
- Utilizes TensorBoard for visualization of epoch-wise training Loss, Validation Loss and mAP50
