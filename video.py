import cv2
import torch

model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')

cam = cv2.VideoCapture("test.MOV")
timestamps = [cam.get(cv2.CAP_PROP_POS_MSEC)]

currentframe = 0
prev = 0
frame_rate = int(cam.get(cv2.CAP_PROP_FPS))
video = cv2.VideoWriter("test.avi", 0, frame_rate, (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while(True):

    ret, frame = cam.read()
    time_elapsed = (cam.get(cv2.CAP_PROP_POS_MSEC) / 1000) - prev

    if not ret:
        break
    
    if time_elapsed > 1.0 / frame_rate:
        curr = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000
        timestamps.append(curr)

        detections = model(frame)
        detections = detections.pandas().xyxy

        for detection in detections:
            if detection.empty:
                break
            tl = (int(detection['xmin'].values[0]), int(detection['ymin'].values[0]))
            br =  (int(detection['xmax'].values[0]), int(detection['ymax'].values[0]))
            label = detection['name'].values[0]
            confidence = detection['confidence'].values[0]
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        video.write(frame)

        print(f"Frame {currentframe}: {detections} Timestamp: {curr}")
        currentframe += 1

        prev = curr

video.release()
cam.release()
cv2.destroyAllWindows()