from djitellopy import tello
import cv2
import numpy as np
import time
from detect_tello import detector
import torch
import numpy as np



x_errors = []
y_errors = []

def chase(pred):
    global x_errors, y_errors
    I_gain = 0
    P_gain = 0.4
    
    person_xyxy_all = [] # 검출된 모든 사람의 bbox 정보가 담길 배열
    
    if len(pred) == 0:
        Tello.send_rc_control(0,0,0,0)
        return
    
    for *xyxy, conf, cls in pred:
        if int(cls)==0:
            person_xyxy_all.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
            
    if len(person_xyxy_all)==0:
        Tello.send_rc_control(0,0,0,0)
        return
    
    # 검출된 사람이 많은 경우, 가장 가까운 사람의 정보만 남김
    person_xyxy = person_xyxy_all[0]
    if len(person_xyxy_all) > 1:
        for i in range(1, len(person_xyxy_all)):
            last_y = person_xyxy[3]
            if last_y > person_xyxy_all[i][3]:
                person_xyxy = person_xyxy_all[i]
    
    x_error = (person_xyxy[0] + person_xyxy[2])/2 - 360
    x_error = x_error
    x_errors.append(x_error)
    if len(x_errors) > 6:
        x_errors.pop(6)
    
    y_error = 390 - person_xyxy[3]
    y_errors.append(y_error)
    if len(y_errors) > 6:
        y_errors.pop(6)
    
    x_control = 0.4*P_gain * x_error + I_gain * sum(x_errors)
    y_control = P_gain * y_error + I_gain * sum(y_errors)
    
    x_control = int(np.clip(x_control, -30, 30))
    y_control = int(np.clip(y_control, -30, 30))
    
    Tello.send_rc_control(0,y_control,0,x_control)


if __name__ == '__main__':
    Tello = tello.Tello()
    Tello.connect()
    print("start battery", Tello.get_battery())
    Tello.streamon()
    time.sleep(5)
    
    initial_img = Tello.get_frame_read().frame
    initial_img = cv2.resize(initial_img,(720,480))
    detector_ = detector(initial_img)
    
    Tello.takeoff()
    time.sleep(3)
    Tello.move_up(20)
    
    while True:
        frame = Tello.get_frame_read().frame
        frame_ = cv2.resize(frame, (720, 480))
        
        with torch.no_grad():
            predictions = detector_.detect(frame_, False)    # True  : 모든 클래스 표시
                                                            # False : 사람만 표시
        chase(predictions)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Tello.land()
            break