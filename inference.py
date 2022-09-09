import torch
model = torch.hub.load("../yolov5", 'custom', path="weights.pt", source='local')
img = './surf1/frame_000000.PNG'
results = model(img)
results.save()