# Mask detection & social distancing detection using YOLOv5 with Intel Realsense D415/D435

# Running Env Setup
Python 3.7 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```
pip install -r requirements.txt
```


## Training @ Virtual Machine
A trained model has saved in `/weights/best.pt`
```bash
python train.py --batch 1 --epochs 200 --data ./data/data.yaml --cfg models/yolov5s.yaml --weights '' --device 0
```


## Inference

* detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
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

To run inference on example video in `data/street.mp4`:

```bash
python detect.py --source data/street.mp4 --weights weights/best.pt
```

To run inference on intel realsense camera:

```bash
python detect.py --source intel --weights weights/best.pt
```

## Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

Thanks to repos from [ultralytics](https://github.com/ultralytics/yolov5) 
and [iAmEthanMai](https://github.com/iAmEthanMai/mask-detection-dataset.git), 
more [ref](https://medium.com/analytics-vidhya/covid-19-face-mask-detection-using-yolov5-8687e5942c81)
