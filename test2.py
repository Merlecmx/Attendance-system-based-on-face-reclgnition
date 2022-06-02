
import os
from skimage import io as iio
import random
import time
import dlib
import numpy as np
import cv2




# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("../model/dlib_face_recognition_resnet_model_v1.dat")
# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')




def return_euclidean_distance(feature_1, feature_2):
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    # print("欧式距离: ", dist)
    if dist > 0.4:
        return False
    else:
        return True


def face_recognition_test(PATH_FACE,batch_size = 256):
    pics = [PATH_FACE +"/" + path + '/' + path +'_0001.jpg' for path in os.listdir(PATH_FACE)]
    random.shuffle(pics)
    start, error_num, not_found_num = 0, 0 , 0
    while start < len(pics):
        end = min(start + batch_size, len(pics))
        paths = pics[start:end]
        start = end
        feature_list = []
        for pic_path in paths:
            img = iio.imread(pic_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = detector(img_gray, 1)
            if len(dets) != 0:
                shape = predictor(img_gray, dets[0])
                face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
                feature_list.append(np.array(face_descriptor))
            else:
                not_found_num += 1
        for i in range(len(feature_list)):
            for j in range(len(feature_list)):
                if i == j and not return_euclidean_distance(feature_list[i], feature_list[j]):
                    error_num +=1
                if i != j and return_euclidean_distance(feature_list[i], feature_list[j]):
                    error_num +=1

    print('found face rate : ', 1.0 - not_found_num / len(pics))
    print('acc: ', 1.0 - error_num / len(pics))




if __name__ == '__main__':
    face_recognition_test('lfw',500)
    print('end')


