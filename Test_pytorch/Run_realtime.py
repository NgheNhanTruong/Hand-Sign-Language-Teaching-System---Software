
import numpy as np
import torch
import cv2
import time
#model_weight = "D:/ALL PROJECTS/VER_4_SOFTWARE_TEACHING_SIGN_LANGUAGE/Model_yoloV5_detection/yolov5/runs/train/exp/weights/best.pt"
#Load model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # model = torch.hub.load('ultarlytics/yolov5', 'custom' , path ='./yolov5/best.pt', source='local')
model = torch.hub.load('ultarlytics/yolov5', 'custom', path ='yolov5/last.pt')
#model = torch.hub.load('./yolov5', 'custom', path="./yolov5/last.pt", source='local') 
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='./last.pt') 
#model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
# model.conf=0.25
frame = cv2.imread('test_2_img.jpg')
# # cap = cv2.VideoCapture(0)
# # while True:
#     # ret,frame=cap.read()
# t=time.time()
# frame=cv2.resize(frame,(640,480))
detections = model(frame)
results = detections.pandas().xyxy[0].to_dict(orient="records")
# x = numpy.array(results)
print(x)

for result in results:
    confidence = result['confidence']
    name = result['name']
    clas = result['class']
    if (clas == 0) or (clas == 1) or (clas == 2) or (clas == 3) or (clas == 4) :
        x1 = int (result['xmin'])
        y1 = int (result['ymin'])
        x2 = int (result['xmax'])
        y2 = int (result['ymax'])
        print(x1,y1,x2,y2)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame, name, (x1+3,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(60,44,255),1)
#print("FPS", 1/(time.time()-t))
cv2.imshow('img',frame)
cv2.waitKey(0)

model = torch.hub.load('D:/ALL PROJECTS/VER_4_SOFTWARE_TEACHING_SIGN_LANGUAGE/Model_yoloV5_detection/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt' ,source='local')

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # Make detections 
#     results = model(frame)
    
#     cv2.imshow('YOLO', np.squeeze(results.render()))
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# # image = cv2.imread('test_2_img.jpg')

