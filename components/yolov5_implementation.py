import torch
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = 'Marlon-Brando-as-Vito-Corleone-in-The-Godfather-Movie.webp' 

results = model(img)
results.show()
results.save()
 
for obj in results.xyxy[0]:
    confidence = obj[4].item() * 100
    label = model.names[int(obj[5])]
    print(f'{label}: {confidence:.2f}%')