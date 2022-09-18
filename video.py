import cv2
import torch

model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')
cam = cv2.VideoCapture("test.MOV")
currentframe = 0
  
while(True):
    ret, frame = cam.read()
  
    if not ret:
        break

    results = model(frame)    
    results.save()
    print(f"Frame {currentframe}: {results.pandas().xyxy}")
    currentframe += 1

cam.release()
cv2.destroyAllWindows()