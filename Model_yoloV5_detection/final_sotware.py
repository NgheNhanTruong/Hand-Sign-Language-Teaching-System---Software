import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from mtcnn.mtcnn import MTCNN
#from sheet import mark_present
import detect_face
from face_aligner import FaceAligner
from PIL import Image
from sklearn.svm import SVC
import pickle
import math
import sys
import cv2
import facenet
import argparse
import numpy as np
from scipy import misc
import tensorflow as tf
from cv2 import phase
from tkinter import image_names
from importlib.resources import path


import cv2
#import mediapipe as mp
import time

def dataset_creation(parameters):
    path1, webcam, face_dim, gpu, username, vid_path = parameters

    path = ''
    res = []
    personNo = 1
    folder_name = ''

    path = path1  # giao diện chọn đường dẫn vào file output

    #############  khởi tạo  #####
    path = 'output'
    res = [640, 480]
    personNo = 1
    folder_name = 'Minh2'
    gpu = 0.8
    face_dim = [160, 160]
    username = 'Minh2'
    vid_path = ''
    #############################

    detector = MTCNN()

    # res = webcam  # giao diện chọn độ phân giải webcam

    gpu_fraction = gpu  # giao diện chọn độ GPU

    # Thông số MTCNN
    minsize = 5
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        gpu_options = tf.GPUOptions()
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    face_size = face_dim  # giao diện chọn độ kích thước khuôn mặt

    affine = FaceAligner(desiredLeftEye=[
                         0, 33, 0, 33], desiredFaceWidth=face_size[0], desiredFaceHeight=face_size[1])

    while True:
        ask = username  # giao diện chọn độ tên folder
        user_folder = path+'/' + folder_name
        image_no = 1
        # giao diện chọn độ tự đông tên file

        data_type = vid_path
        loop_type = False
        total_frames = 0
        if data_type == '':
            data_type = 0
            loop_type = True

        # Initialize webcam or video
        device = cv2.VideoCapture(-2)
        # If webcam set resolution
        if data_type == 0:
            device.set(3, res[0])
            device.set(4, res[1])
        # else:
        #     #Finding total number of frames of video
        #     total_frames=int(device.get(cv2.CAP_PROP_FRAME_COUNT))
        #     print('tổng frame', total_frames)
        #     #Shutting down web cam variable
        #     loop_type=False

        # Start web cam or start video and start creating dataset by user.
        while loop_type or (total_frames > 0):
            # Nếu dùng thêm video cần...

            ret, image = device.read()
            #cv2.waitKey(5)

            # Run MTCNN and do face detection until 's' keyword is pressed
            if (cv2.waitKey(1) & 0xFF) == ord("s"):
                #bb, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
                detect = detector.detect_faces(image)
                print(detect)

                if detect:
                    bb = detect[0]['box']
                    print('bounding box', bb)
                    x, y, w, h = bb

                    aligned_image = image[y:y+h, x:x+w]
                    # aligned_image=affine.align(image,points)

                    image_name = user_folder+'/'+folder_name + \
                        '_'+str(image_no).zfill(4)+".png"
                    print('Path1', image_name)
                    print('Path2', user_folder)
                    cv2.imwrite(image_name, aligned_image)
                    image_no = image_no+1

                    image=cv2.rectangle(image,(x,y), (x+w,y+h),color=(0, 255, 0),thickness=2)

                    # img=cv2.imread(image_name)
                    # cv2.imshow('Anh 2',img)
                    # cv2.waitKey(1)

                    """ # for i in range(bb.shape[0]):
                    #     cv2.rectangle(image, (int(bb[i][0]), int(bb[i][1])), (int(
                    #         bb[i][2]), int(bb[i][3])), (0, 255, 0), 2)
                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw each of them
                    # for col in range(points.shape[1]):
                    #     for i in range(5):
                    #         cv2.circle(image, (int(points[i][col]), int(points[i + 5][col])), 1, (0, 255, 0), -1) """
                    # cv2.waitKey(5)

            # Show the output video to user
            cv2.imshow("Output", image)

            # Break this loop if 'q' keyword pressed to go to next user
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                device.release()
                cv2.destroyAllWindows()

                # break
                abcd = 1
                return abcd


def train(parameters):
    path1, path2, batch, img_dim, gpu, svm_name, split_percent, split_data = parameters

    #############  khởi tạo  #####
    path1 = 'output'
    path2 = 'model/20180402-114759-CASIA-WebFace/20180402-114759.pb'
    gpu = 0.8
    split_data = 'Y'
    svm_name = 'classifier.pkl'
    #############################

    path = path1  # giao diện nhập dường dẫn hiện tại

    gpu_fraction = gpu  # giao diện chọn GPU

    model = path2  # giao diện chọn đường dẫn cho model

    batch_size = 90  # giao diện chọn batch size

    image_size = 160

    ask = img_dim  # giao diện

    classifier_filename = svm_name  # giao diện chọn trên file classifier

    split_dataset = split_data
    percentage = 20  # giao diện

    min_nrof_images_per_class = 0
    dataset = facenet.get_dataset(path)  # file của từng cá nhân
    train_set = []
    test_set = []

    if split_dataset == 'Y':
        for cls in dataset:
            paths = cls.image_paths  # đường dẫn của từng file ảnh
            # Remove classes with less than min_nrof_images_per_class
            if len(paths) >= min_nrof_images_per_class:
                # np.random.shuffle(paths)

                # Find the number of images in training set and testing set for this class
                no_train_images = int(percentage*len(paths)*0.01)

                train_set.append(facenet.ImageClass(
                    cls.name, paths[:no_train_images]))  # ???
                test_set.append(facenet.ImageClass(
                    cls.name, paths[no_train_images:]))  # ???

    paths_train = []
    labels_train = []
    paths_test = []
    labels_test = []
    emb_array = []
    class_names = []

    if split_dataset == 'Y':
        paths_train, labels_train = facenet.get_image_paths_and_labels(
            train_set)
        paths_test, labels_test = facenet.get_image_paths_and_labels(test_set)
        print('\nNumber of classes: %d', len(train_set))  # 2
        print('\nNumber of images in TRAIN set: %d', len(
            paths_train))  # So luong file anh train 126
        print('\nNumber of images in TEST set: %d', len(
            paths_test))  # So luong file anh test 56
    else:
        paths_train, labels_train = facenet.get_image_paths_and_labels(dataset)
        print('\nNumber of classes: %d', len(dataset))
        # paths_train la tat ca cac duong dan cua anh
        print('\nNumber of images in TRAIN set: %d', len(paths_train))
        print('\nNumber of images in TEST set: %d', len(paths_test))

    # Find embedding
    emb_array = get_embeddings(
        model, paths_train, batch_size, image_size, gpu_fraction)

    # Train the classifier
    print('\n Training classifier')
    model_svc = SVC(kernel='linear', probability=True)
    model_svc.fit(emb_array, labels_train)

    # Create a list of class names
    if split_dataset == 'Y':
        class_names = [cls.name.replace('_', ' ') for cls in train_set]
    else:
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

    # Saving classifier model
    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model_svc, class_names), outfile)
    print('\nSaved classifier model to file: "%s"', classifier_filename)

    if split_dataset == 'Y':
        # Find embedding for test data
        emb_array = get_embeddings(
            model, paths_train, batch_size, image_size, gpu_fraction)

        # Call test on the test set
        parameters = '', '', '', '', '', gpu_fraction
        test(parameters, classifier_filename, emb_array,
             labels_test, model, batch_size, image_size)
    c = 1
    return c


def test(parameters, classifier_filename, emb_array, labels_test, model, batch_size, image_size):

    print('Hello')


def get_embeddings(model, paths, batch_size, image_size, gpu_fraction):
    # initializing the facenet tensorflow model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # Load the model
            print('\nLoading feature extraction model')
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')  # ???
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')  # ???
            phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name('phase_train:0')  # ???
            embedding_size = embeddings.get_shape()[1]
            print('Kiem tra type embeddings ', type(embeddings))

            # Run forward pass to calculate embedding
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(
                math.ceil(1.0*nrof_images/batch_size))  # Iteration
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                print(paths_batch)

                # Does random crop, prewhitening and flipping
                #images=facenet.load_data(paths_batch, False, False, image_size)
                images = facenet.load_data(paths_batch, True, True, image_size)

                # Get the embeddings
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}  # ???
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)  # ???

    return emb_array


def recognize(mode, parameters):
    print(parameters)
    path1, path2, face_dim, gpu, thresh1, thresh2, resolution, img_path, out_img_path, vide_path, vid_save, vid_see = parameters
    st_name = ''

    # classifier_filename = path1
    # model = path2
    # ask = face_dim
    # gpu_fraction = gpu

    ### Khởi tạo ###
    classifier_filename = 'classifier.pkl'
    model = "20180402-114759/20180402-114759.pb"
    image_size = (160, 160)
    gpu_fraction = 0.8
    mode = 'w'
    time_taken_1 = []
    time_taken = []
    ################

    # Taking the parameters for recoignition by the user
    # giao diện

    # input_type = input("\nPress I for image input OR\nPress V for video input OR\nPress W for webcam input OR\nPress ENTER for default webcam: ").lstrip().rstrip().lower()
    # if input_type == "":
    #  input_type = 'w'
    input_type = mode

    # Load the face aligner model
    affine = FaceAligner(desiredLeftEye=(
        0.33, 0.33), desiredFaceWidth=image_size[0], desiredFaceHeight=image_size[1])

    # Building seperate graphs for both the tf architectures
    g1 = tf.Graph()
    g2 = tf.Graph()

    # Load the model for FaceNet image recognition
    with g1.as_default():
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with tf.compat.v1.Session() as sess:
            facenet.load_model(model)

    # Load the model of MTCNN face detection.
    with g2.as_default():
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    # Some MTCNN network parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.8]  # three steps' threshold
    factor = 0.709  # scale factor

    # giao diện

    classifier_threshold = 0.50

    # loading the classifier model
    with open(classifier_filename, 'rb') as infile:
        (modelSVM, class_names) = pickle.load(infile)

    # helper variables
    image = []
    device = []
    display_output = True

    # Webcam variables
    loop_type = False
    res = (630, 480)

    # Video input variables
    total_frames = 0
    save_video = False
    frame_no = 1
    output_video = []

    # Image input type variables
    save_images = False
    image_folder = ''
    out_img_folder = ''
    imageNo = 1
    image_list = []
    image_name = ''

    # if webcam is selected
    if input_type == 'w':
        data_type = 0
        loop_type = True
        # giao diện chon resolution

    # Initialize webcam or video if no image format
    if input_type != "i":
        device = cv2.VideoCapture(data_type)

    # If webcam set resolution
    if input_type == 'w':
        device.set(3, res[0])
        device.set(4, res[1])  # ???

    # elif input_type =='v':

    while loop_type or (frame_no <= total_frames):
        if input_type == 'i':
            image = cv2.imread(image_folder+'/' + image_list[frame_no-1])
        else:
            ret, image = device.read()

        # ### Tính FPS 1a ###
        # start_time_1 = time.time()
        ### Tính FPS 2a ###
            start_time = time.time()

        # Run MTCNN model to detect faces
        # g2.as_default()
        # with tf.Session(graph=g2) as sess:
        #     # we get the bouding boxes as well as the points for the faces
        #     frame = image
        #     # cv2.imshow('Kiem tra frame anh tu Webcam',frame)
        #     # /home/ml/Documents/attendance_dl/dataset/test.mp4
        #     image = cv2.resize(image, (800, 600))

        #     # Không hiểu tại sao lại dùng
        #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #     value = 0
        #     h, s, v = cv2.split(hsv)
        #     v = v-value
        #     #h -= value
        #     image = cv2.merge((h, s, v))
        #     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        #     ###############################################################

        #     #image = noisy('speckle', image)
        #     # chuyển ảnh vào thành một mảng
        #     image = np.asarray(image, dtype='uint8')
            # bb, points = detect_face.detect_face(
            #     image, minsize, pnet, rnet, onet, threshold, factor)
        #     # print("Loai bb",type(bb))
        #     # print("Loai poijt",type(points))

        ###################
        mp_facedetector = mp.solutions.face_detection
        mp_draw = mp.solutions.drawing_utils
        with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
                # Convert the BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image and find faces
                results = face_detection.process(image)
                
                # Convert the image color back so it can be displayed
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.detections:
                    for id, detection in enumerate(results.detections):
                        mp_draw.draw_detection(image, detection)
                        #=list(detection)
                        #print(detection)
                        #points=detection
                        bb = detection.location_data.relative_bounding_box
                        
                        x0=detection.location_data.relative_keypoints[0].x
                        x1=detection.location_data.relative_keypoints[1].x
                        x2=detection.location_data.relative_keypoints[2].x
                        x3=detection.location_data.relative_keypoints[3].x
                        x4=detection.location_data.relative_keypoints[4].x
                        x5=detection.location_data.relative_keypoints[5].x

                        y0=detection.location_data.relative_keypoints[0].y
                        y1=detection.location_data.relative_keypoints[1].y
                        y2=detection.location_data.relative_keypoints[2].y
                        y3=detection.location_data.relative_keypoints[3].y
                        y4=detection.location_data.relative_keypoints[4].y
                        y5=detection.location_data.relative_keypoints[5].y

                        #point=point1[:,1]
                        print(point1)
                        #print('Loai bb',list(bb))

        # ### Tính FPS 2b ###
        # total_time_1 = time.time() - start_time_1
        # time_taken_1.append(total_time_1)
        # mean_time_1 = np.mean(time_taken_1)
        # mean_fps_1 = 1/mean_time_1
        # print(
        #     f"___Mang MTCNN___ Mean Time: {mean_time_1:1.7f} - Mean FPS: {mean_fps_1:1.7f}")

        # See if face is detected
        # if bb.shape[0] > 0:
        #     # ALIGNMENT - use the bounding boxes and facial landmarks points to align images

        #     # create a numpy array to feed the network
        #     img_list = []
        #     # Trả mảng có kích thức 1,600,800 về 0
        #     images = np.empty([bb.shape[0], image.shape[0], image.shape[1]])

        #     for col in range(points.shape[1]):
        #         # Chuyển từ ma trận 10x1 thành 1x10 #Giảm kích thước ảnh về 160x160
        #         aligned_image = affine.align(image, points[:, col])
        #         # print(aligned_image)
        #         # print(str(len(aligned_image)))

        #         # Prewhiten the image for the facenet architecture to give better results
        #         # Cân bằng diagram (Hiểu)
        #         mean = np.mean(aligned_image)  # tính trung bình cộng
        #         std = np.std(aligned_image)  # tính độ lệch tiêu chuẩn
        #         std_adj = np.maximum(
        #             std, 1.0 / np.sqrt(aligned_image.size))  # có chỉnh sửa
        #         ready_image = np.multiply(
        #             np.subtract(aligned_image, mean), 1 / std_adj)
        #         img_list.append(ready_image)
        #         images = np.stack(img_list)  # Hình dạng 1x160x160x3 ###???
        #         # print(images.shape)

        #     # ### Tính FPS 2a ###
        #     # start_time = time.time()

        #     # EMBEDDINGS: Use the processed aligned images for Facenet embeddings

        #     g1.as_default()
        #     with tf.Session(graph=g1) as sess:
        #         # Run forward pass on FaceNet tp get the ambeddings
        #         images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        #         embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        #         feed_dict = {images_placeholder: images,
        #                      phase_train_placeholder: False}
        #         # print(feed_dict)
        #         embedding = sess.run(embeddings, feed_dict=feed_dict)

        #     # PREDICTION: use the classifier to predict the most likely class (person).
        #     predictions = modelSVM.predict_proba(embedding)
        #     best_class_indices = np.argmax(predictions, axis=1)
        #     best_class_probabilities = predictions[np.arange(
        #         len(best_class_indices)), best_class_indices]

        #     ### Tính FPS 2b ###
        #     total_time = time.time() - start_time
        #     time_taken.append(total_time)
        #     mean_time = np.mean(time_taken)
        #     mean_fps = 1/mean_time
        #     print(f"___Mang Deep___ Mean Time: {mean_time:1.7f} - Mean FPS: {mean_fps:1.7f}")

        # # DRAW: draw bounding boxes, landmarks and predicted names
        # for i in range(bb.shape[0]):
        #     cv2.rectangle(image, (int(bb[i][0]), int(bb[i][1])), (int(
        #         bb[i][2]), int(bb[i][3])), (0, 255, 0), 1)

        #     # Put name and probability of detection only if given threshold is crossed
        #     if best_class_probabilities[i] > classifier_threshold:
        #         cv2.putText(image, class_names[best_class_indices[i]], (int(bb[i][0] + 1), int(
        #             bb[i][1]) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        #         # print(class_names[best_class_indices[i]])
        #         st_name += ','
        #         st_name += class_names[best_class_indices[i]]
        #         mark_present(st_name)
        #         # cv2.waitKey(0)
        #         #cv2.putText(image, str(round(best_class_probabilities[i] * 100, 2)) + "%", (int(bb[i][0]), int(bb[i][3]) + 7), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Out', image)

        if cv2.waitKey(10) == 'q':
            # do a bit of cleanup
            # if save_video:
            #     output_video.release()
            device.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    dataset_creation([0, 0, 0, 0, 0, 0])
    # train([0, 0, 0, 0, 0, 0, 0, 0])
    #recognize('', ['', '', '', '', '', '', '', '', '', '', '', ''])
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
