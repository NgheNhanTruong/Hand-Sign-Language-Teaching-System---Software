import cv2
import os
import time
import uuid

IMAGES_PATH = 'D:\ALL PROJECTS\VER_3_SOFTWARE_TEACHING_SIGN_LANGUAGE\Object_Detection_Ver_3\Hinh_Origin\A'


labels = ['A']
number_imgs = 20

for label in labels:
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    print ("Collecting images for {}".format(label))
    time.sleep(5)
    for imgnum in range (number_imgs):
        ret,frame = cap.read()
        imagename = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
print("DONE")