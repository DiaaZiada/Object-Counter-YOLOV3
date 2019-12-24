# Object Counter (YOLO_V3)
![YOLO](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/yolo.jpg)

Object counter is a toolkit that uses YOLO V3(you only look once version 3) algorithm. It makes an object detection on images/videos and count the number of objects present in the image/video.

[this video test the toolkit on part of video of captain marvel trailer](https://l.facebook.com/l.php?u=https://www.youtube.com/watch?v=r1SkEeA2nzw&feature=youtu.be&fbclid=IwAR31CfTqv3LopKjU--m8FDexXbtk2NjaZgCOWnL9Xwp5YWzjoHLNbMVTN2A&h=AT1wabibFHkTwEX7gAZhlZ8hcE_dBt_bH8_xVE81h-iMDV6hY7gY_yBJ2chCvWwaQYx9BbfACBGJJJLu5iWi7vmvgmpmJaj5bCz0okxk0ZMjaTV82FtbQfkjWK_b7A)
and here some examples of prediction on images 
![  ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/eagle.jpg)
![  ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/herd_of_horses.jpg)
![  ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/img1.jpg)
![  ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/img2.jpg)
![ ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/img3.jpg)
![ ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/img4.jpg)
![ ](https://github.com/DiaaZiada/Object-Counter-YOLOV3/blob/master/github_images/person.jpg)

this toolkit allow you to make predictions using pre-trained weights or train the network in your own dataset for making predictions.
if you have to count a lot of objects daily like goods you can use this toolkit to count for your in less than second.

****Requirements****
	

 - [Python](https://www.python.org/) 3.*
 - [Numpy](http://www.numpy.org/)
 - [OpenCV](https://opencv.org/)
 - [Pytorch](https://pytorch.org/)
 
 there are three cases to run this toolkit:
 
 - make predictions on images
 - make predictions on videos
 - train the model on custom dataset
 
 ## Setup For Predictions
 After download and extract the repository
 
 #### Weights
Add your weights in weights directory, or download coco weights for yolov3 of tiny yolo
by running these commands
yolo -> `$ sh weights/download_weights.sh`
tiny yolo -> `$ sh weights/download_weights_tiny.sh`

#### Configurations
Add your configuration file for your model in config directory, or use yolov3 of tiny yolo configurations models files
 
## Images
`images.py` is the script that is responsible for managing predictions on images
```
usage: images.py [-h] [--images IMAGES] [--outputs OUTPUTS] [--bs BS]
                 [--confidence CONFIDENCE] [--nms_thresh NMS_THRESH]
                 [--cfg CFGFILE] [--weights WEIGHTSFILE] [--reso RESO]

YOLO v3 detection Module

optional arguments:
  -h, --help            show this help message and exit
  --images IMAGES       Image / Directory containing images to perform
                        detection upon
  --outputs OUTPUTS     Image / Directory to store detections
  --bs BS               Batch size
  --confidence CONFIDENCE
                        Object Confidence to filter predictions
  --nms_thresh NMS_THRESH
                        NMS Threshhold
  --cfg CFGFILE         Config file
  --weights WEIGHTSFILE
                        weightsfile
  --reso RESO           Input resolution of the network. Increase to increase
                        accuracy. Decrease to increase speed

```
## Videos
`video.py` is the script that is responsible for managing predictions on videos
```
usage: video.py [-h] [--bs BS] [--confidence CONFIDENCE]
                [--nms_thresh NMS_THRESH] [--config CONFIGFILE]
                [--weights WEIGHTSFILE] [--outputs OUTPUTS] [--reso RESO]
                [--video VIDEOFILE] [--cam CAMERA]

YOLO v3 Detection Module

optional arguments:
  -h, --help            show this help message and exit
  --confidence CONFIDENCE
                        Object Confidence to filter predictions
  --nms_thresh NMS_THRESH
                        NMS Threshhold
  --config CONFIGFILE   Config file
  --weights WEIGHTSFILE
                        weightsfile
  --outputs OUTPUTS     Image / Directory to store detections
  --reso RESO           Input resolution of the network. Increase to increase
                        accuracy. Decrease to increase speed
  --video VIDEOFILE     Video file to run detection on
  --cam CAMERA          use camera to make detections

```
## Setup For Training 

After downloading and extracting the repository
**COCO DataSet**
	For downloading the COCO dataset run this command
	`$  sh data/get_coco_dataset.sh `
**Custom DataSet**
	Get into data directory and put our dataset in this directory (to manage the files)
	the dataset structure must be like coco data set 
 - images: should be in directory named **images** 
 - labels : should be in directory named **labels**, the label of each image must has same name but the extinction be `.txt` 
 - data configuration file should be in **config** directory and be like this
	 ```
	classes=80
	train=data/coco/trainvalno5k.txt
	valid=data/coco/5k.txt
	names=data/coco.names
	backup=backup/
	eval=coco
	```

## Train
`train.py` is the script that is responsible for managing the training process
```
usage: train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR] [--use_cuda USE_CUDA]
                [--weights WEIGHTSFILE]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs
  --image_folder IMAGE_FOLDER
                        path to dataset
  --batch_size BATCH_SIZE
                        size of each image batch
  --model_config_path MODEL_CONFIG_PATH
                        path to model config file
  --data_config_path DATA_CONFIG_PATH
                        path to data config file
  --weights_path WEIGHTS_PATH
                        path to weights file
  --class_path CLASS_PATH
                        path to class label file
  --conf_thres CONF_THRES
                        object confidence threshold
  --nms_thres NMS_THRES
                        iou thresshold for non-maximum suppression
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --img_size IMG_SIZE   size of each image dimension
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between saving model weights
  --checkpoint_dir CHECKPOINT_DIR
                        directory where model checkpoints are saved
  --use_cuda USE_CUDA   whether to use cuda if available
```

 ## Credits
 - [YOLO v3 tutorial from scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)
 - [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
