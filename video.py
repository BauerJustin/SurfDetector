from time import time
import cv2
import torch

model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')

cam = cv2.VideoCapture("test.MOV")
timestamps = [cam.get(cv2.CAP_PROP_POS_MSEC)]

currentframe = 0
  
while(True):
    ret, frame = cam.read()
  
    if not ret:
        break
    
    timestamps.append(cam.get(cv2.CAP_PROP_POS_MSEC))

    results = model(frame)    
    results.save()
    print(f"Frame {currentframe}: {results.pandas().xyxy} Timestamp: {timestamps[-1]}")
    currentframe += 1

cam.release()
cv2.destroyAllWindows()