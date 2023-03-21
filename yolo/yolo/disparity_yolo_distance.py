import cv2
import torch
from numpy import random
import numpy as np
import os

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid

from cv_bridge import CvBridge
import message_filters

PATH = os.path.dirname(__file__)

class Detector(Node):
    def __init__(self):
        ####################### ros init
        super().__init__('yolo')
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 1)
        self.image1_sub = message_filters.Subscriber(self, Image, "image_raw1")
        self.image2_sub = message_filters.Subscriber(self, Image, "image_raw2")
        self.synchronize = message_filters.TimeSynchronizer([self.image1_sub, self.image2_sub], 1)
        self.synchronize.registerCallback(self.image_callback)
        
        self.bridge = CvBridge()

        window_size = 3
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=-10,numDisparities = 160,
                                  blockSize = 23,
                                  P1=5*3*window_size **2,
                                  P2=8*3*window_size **2,
                                  disp12MaxDiff = -1,
                                  uniquenessRatio = 15,
                                  speckleWindowSize=0,
                                  speckleRange=2,
                                  preFilterCap=63,
                                  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        lmbda = 80000
        sigma = 1.2

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = self.left_matcher)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)

        ############################# yolo init
        self.imgsz = 736
        # Initialize
        set_logging()
        self.device = torch.device('cuda')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        self.weights = PATH + '/yolov7.pt'
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

        img = np.zeros((640,640,3))

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
        print("yolo init")
 
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
                    if display_all:
                        predict_result.append([xyxy,conf,cls])
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    else:
                        if int(cls) == 0:
                            predict_result.append([xyxy,conf,cls])
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            # print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # Stream results     
        cv2.imshow("result", im0)
        cv2.waitKey(1)  # 1 millisecond
        return predict_result
    
    def image_callback(self, msg1, msg2):
        imageL = self.bridge.imgmsg_to_cv2(msg1, 'bgr8')
        img = imageL.copy()
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
        imageR = self.bridge.imgmsg_to_cv2(msg2, 'bgr8')
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

        disp_left = self.left_matcher.compute(imageL, imageR)
        disp_right = self.right_matcher.compute(imageR, imageL)
        disp_left = np.int16(disp_left)
        disp_right = np.int16(disp_right)
        filteredImg = self.wls_filter.filter(disp_left,imageL,None,disp_right)

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg,
                                    beta = 0, alpha = 255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)

        detection_result = self.detect(img, False)

        self.object_visualize(detection_result, filteredImg)
        cv2.imshow("st_Disparity Map",filteredImg)
        cv2.waitKey(1)

        # self.detect(image, True)
    def object_visualize(self, detections, depthmap):
        
        map_height = 600
        map_width = 720
        # nav_msgs/OccupancyGrid Message 선언
        map = OccupancyGrid()
        map.info.resolution = 0.02
        map.info.width = map_width
        map.info.height = map_height
        map.header.frame_id = '/map'
        map.info.origin.position.x = -map_width*0.02/2
        map.info.origin.position.y = -map_height*0.02/2

        # numpy map 생성
        map_numpy = np.zeros((map_height, map_width), np.int8)
        image_width = depthmap.shape[1]
        for xyxy, conf, cls in detections:
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            depthmap_avg = np.average(depthmap[ int(xyxy[1]+h*2/5):int(xyxy[3]-h*2/5),
                                                int(xyxy[0]+w*2/5):int(xyxy[2]-w*2/5)])
            
            distance = depthmap_avg * 2
            center = ((xyxy[2] + xyxy[0])/2) - image_width/2
            center = int(map_width/2 + (center * distance / 300))
            
            print(distance, "     ", center)
            cv2.circle(map_numpy, (center, int(distance)), 20, 100, -1)
        map_numpy = cv2.flip(map_numpy, 1)
        
        map.data = map_numpy.reshape(-1).tolist()
        self.map_pub.publish(map)
        cv2.imshow("map", map_numpy)


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt :
        node.get_logger().info('Stopped by Keyboard')
    finally :
        node.destroy_node()

if __name__ == "__main__":
    main()
