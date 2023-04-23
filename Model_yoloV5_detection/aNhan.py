import cv2
from matplotlib import image
import torch


def yolov5(image):
    model_sign.conf = 0.7  # confidence threshold (0-1)
    model_sign.iou = 0.7  # NMS IoU threshold (0-1)
    result = model_sign(image, size = 640)
    result.display(render=True) 
    return result

if __name__ == '__main__':
    model_sign = torch.hub.load('ultralytics/yolov5', 'custom', path='checkpoints/traffic/bestv7.pt')
    image = ...
    imagefbbox = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = yolov5(imagefbbox)
    print(result.pandas().xyxy[0].name[0])