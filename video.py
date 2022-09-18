import cv2
import torch

frame_rate = 10

model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')

cam = cv2.VideoCapture("test.MOV")
timestamps = [cam.get(cv2.CAP_PROP_POS_MSEC)]

currentframe = 0
prev = 0

while(True):

    ret, frame = cam.read()
    time_elapsed = (cam.get(cv2.CAP_PROP_POS_MSEC) / 1000) - prev

    if not ret:
        break
    
    if time_elapsed > 1.0 / frame_rate:
        curr = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000
        timestamps.append(curr)

        results = model(frame)    
        
        results.save()
        print(f"Frame {currentframe}: {results.pandas().xyxy} Timestamp: {curr}")
        currentframe += 1

        prev = curr

cam.release()
cv2.destroyAllWindows()