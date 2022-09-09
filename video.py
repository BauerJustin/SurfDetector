# Importing all necessary libraries
import cv2
import os
import torch

# Read the video from specified path
cam = cv2.VideoCapture("test.MOV")

model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')

try:
      
    # creating a folder named data
    if not os.path.exists('frames'):
        os.makedirs('frames')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of frames')
  
# frame
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = './frames/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)
  
        # writing the extracted images
        results = model(frame)
        results.save()
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()