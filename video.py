from __future__ import division
import time
from collections import Counter
import argparse
import os 

import pickle as pkl
import torch 
from torch.autograd import Variable
import cv2 

from util.parser import load_classes
from util.model import Darknet
from util.image_processor import prep_image
from util.utils import non_max_suppression


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--config", dest = 'configfile', help =  "Config file", default = "config/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "weights/yolov3.weights", type = str)
    parser.add_argument("--outputs", dest = 'outputs', help = "Image / Directory to store detections to", default = "outputs", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", type = str)
    parser.add_argument("--cam", dest = "camera", help = "use camera to make detections", type = bool)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.configfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()
obj_counter = {}

#Make output dir if it's not exist
if not os.path.exists(args.outputs):
    os.makedirs(args.outputs)


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = colors[cls%100]
    label = "{0}: {1}".format(classes[cls],str(obj_counter[cls]))
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase
if args.videofile:
    videofile = args.videofile #or path to the video file. 
    video_path = os.path.join(args.outputs,videofile.split('/')[0].split('\\')[-1])
    cap = cv2.VideoCapture(videofile)  
    assert cap.isOpened(), 'Cannot capture source'

elif args.camera:    
    cap = cv2.VideoCapture(0)
    video_path = os.path.join(args.outputs,'cam_output.mp4')
else:
    raise "there is not video or camera option choosen, please chose one option to start work on it"
    



frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
frames = 0  
start = time.time()

while True:
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True))
        output = non_max_suppression(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        
        obj_counter = output[:,-1]
        obj_counter = Counter(obj_counter.numpy().astype(int).tolist())


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("colors/pallete", "rb"))
        clss = {}
        list(map(lambda x: write(x, frame), output))
        out.write(frame)
        
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or not cap.isOpened() or not cap.isOpened():
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     

# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 







