
# Importing all necessary libraries
import cv2
import os
  
# Read the video from specified path
cam = cv2.VideoCapture("D:\ALL PROJECTS\VER_4_SOFTWARE_TEACHING_SIGN_LANGUAGE\Model_yoloV5_detection\Videos\Test_video_results.mp4")
link = "D:\ALL PROJECTS\VER_4_SOFTWARE_TEACHING_SIGN_LANGUAGE\Model_yoloV5_detection\Result_yolo_separated" 
# try:
      
#     # creating a folder named data
#     if not os.path.exists('data_1'):
#         os.makedirs('data_1')
  
# # if not created then raise error
# except OSError:
#     print ('Error: Creating directory of data')
  
# frame
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        
        if currentframe%15 == 0:
            # if video is still left continue creating images
            name = str(link)+ "/" + str(currentframe) + '.jpg'
            print ('Creating..' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1
        print(frame.shape)
    else:
        
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()