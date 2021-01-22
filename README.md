# Combination of two repo

git clone https://github.com/ultralytics/yolov5

git clone https://github.com/iAmEthanMai/mask-detection-dataset.git

# Yolo v5
https://medium.com/analytics-vidhya/covid-19-face-mask-detection-using-yolov5-8687e5942c81

Python interpreter: Python 3.7 (SCAN)


### For yolov5 env setup:

pip install -U -r requirements.txt


### Move VM files to local

scp tiliter@52.249.194.76:yolov5/runs/train/exp2/weights/best.pt .

scp -r tiliter@52.249.194.76:yolov5/runs/train/exp3 .


### @VM
* Training

python train.py --batch 1 --epochs 200 --data ../mask-detection-dataset/data/data.yaml --cfg models/yolov5s.yaml --weights '' --device 0
* Testing 

python detect.py --source ../mask-detection-dataset/data/images/maksssksksss787.png --weights runs/train/exp2/weights/best.pt


### @local Mac
* Testing 

python detect.py --source data/street.mp4 --weights runs/train/exp2/weights/best.pt



## Pretrained Checkpoints

| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>V100</sub> | FPS<sub>V100</sub> || params | GFLOPS |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    |640 |36.8     |36.8     |55.6     |**2.2ms** |**455** ||7.3M   |17.0
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    |640 |44.5     |44.5     |63.1     |2.9ms     |345     ||21.4M  |51.3
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| | | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA |832 |**51.9** |**51.9** |**69.6** |24.9ms |40      ||87.7M  |1005.3

<!--- 
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |640 |49.0     |49.0     |67.4     |4.1ms     |244     ||77.2M  |117.7
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |1280 |53.0     |53.0     |70.8     |12.3ms     |81     ||77.2M  |117.7
--->

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or TTA. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
** Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes image preprocessing, FP16 inference, postprocessing and NMS. NMS is 1-2ms/img.  **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce TTA** by `python test.py --data coco.yaml --img 832 --iou 0.65 --augment` 


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)


## Inference

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', img_size=640, iou_thres=0.45, save_conf=False, save_dir='runs/detect', save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
Using torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16130MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt to yolov5s.pt... 100%|██████████████| 14.5M/14.5M [00:00<00:00, 21.3MB/s]

Fusing layers... 
Model Summary: 232 layers, 7459581 parameters, 0 gradients
image 1/2 data/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.012s)
image 2/2 data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.012s)
Results saved to runs/detect/exp
Done. (0.113s)
```
<img src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg" width="500">  

### PyTorch Hub

To run **batched inference** with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36):
```python
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img1 = Image.open('zidane.jpg')
img2 = Image.open('bus.jpg')
imgs = [img1, img2]  # batched list of images

# Inference
result = model(imgs)
```


## Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 
