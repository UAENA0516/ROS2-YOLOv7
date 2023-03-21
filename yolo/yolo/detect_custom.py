import cv2
import torch
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox

class detector():
    def __init__(self, img):
        self.imgsz = 736
        # Initialize
        set_logging()
        self.device = torch.device('cuda')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.weights = 'yolov7.pt'
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        img = [letterbox(img, self.imgsz, auto=True, stride=self.stride)[0]]
        img = np.stack(img, 0)

            # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]
 
    def detect(self, img, display_all):
        im0 = img.copy()
        
        img = [letterbox(img, self.imgsz, auto=True, stride=self.stride)[0]]
        img = np.stack(img, 0)

            # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        conf_thres = 0.25
        iou_thres = 0.45

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        predict_result = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    predict_result.append([*xyxy,conf,cls])
                    if display_all:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    else:
                        if int(cls) == 0:
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            # print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # Stream results     
        cv2.imshow("result", im0)
        cv2.waitKey(1)  # 1 millisecond 
        
        return predict_result

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    _, frame = cap.read()
    detector_ = detector(frame)
    
    while True:
        _, frame = cap.read()
        frame_ = cv2.resize(frame, (736,640))
        with torch.no_grad():
            detector_.detect(frame_, True)