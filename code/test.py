
import os
from skimage import io as iio
import random
import time
import dlib
import numpy as np
import cv2
import pymysql



# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("../model/dlib_face_recognition_resnet_model_v1.dat")
# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')

def is_same_face(feature_1, feature_2):
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    # print("欧式距离: ", dist)
    if dist > 0.4:
        return False
    else:
        return True
def face_recognition_test(path_face,batch_size = 256):
    pics = [path_face +"/" + path + '/' + path +'_0001.jpg' for path in os.listdir(path_face)]
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
                if (i == j and not is_same_face(feature_list[i], feature_list[j]) ) or \
                        ( i != j and is_same_face(feature_list[i], feature_list[j]) ):
                    error_num +=1

    print('found face rate : ', 1.0 - not_found_num / len(pics))
    print('acc: ', 1.0 - error_num / len(pics))

def adapt_array(arr):
    return np.array(arr).tostring()

def convert_array(text):
    return np.frombuffer(text, dtype=np.float64)

def insertARow(Row, type):
    conn = pymysql.connect(host='139.196.146.45', user='root', password='Iotlab2019@217', database='face')
    # conn = sqlite3.connect("inspurer.db")  # 建立数据库连接
    cur = conn.cursor()  # 得到游标对象
    if type == 1:
        cur.execute("insert into worker_info (id,name,face_feature) values(%s,%s,%s)",
                    ([Row[0], Row[1], adapt_array(Row[2])]))
        print("写人脸数据成功")
    if type == 2:
        print(Row)
        cur.execute("insert into logcat (id,name,datetime,late) values(%s,%s,%s,%s)",
                    ([Row[0], Row[1], Row[2], Row[3]]))
        print("写日志成功")
        pass
    cur.close()
    conn.commit()
    conn.close()

def loadDataBase(type):

    conn = pymysql.connect(host='139.196.146.45', user='root', password='Iotlab2019@217', database='face')
    cur = conn.cursor()  # 得到游标对象

    if type == 1:
        knew_id = []
        knew_name = []
        knew_face_feature = []
        cur.execute('select id,name,face_feature from worker_info')
        origin = cur.fetchall()
        for row in origin:
            print(row[0])
            knew_id.append(row[0])
            print(row[1])
            knew_name.append(row[1])
            # print(self.convert_array(row[2]))
            print('row[2]: ',convert_array(row[2]).shape)
            knew_face_feature.append(convert_array(row[2]))
    if type == 2:
        logcat_id = []
        logcat_name = []
        logcat_datetime = []
        logcat_late = []
        cur.execute('select id,name,datetime,late from logcat')
        origin = cur.fetchall()
        for row in origin:
            print(row[0])
            logcat_id.append(row[0])
            print(row[1])
            logcat_name.append(row[1])
            print(row[2])
            logcat_datetime.append(row[2])
            print(row[3])
            logcat_late.append(row[3])
    pass

def db_test():
    # 写入数据库测试
    start_time = time.time()
    for i in range(100):
        feature = np.random.randn(128)
        row = [i, 'name'+ str(i), feature]
        insertARow(row, 1)
    end_time = time.time()
    print('run time: ', end_time - start_time)

    # 数据库读取测试
    start_time = time.time()
    for i in range(100):
        feature = np.random.randn(128)
        row = [i, 'name' + str(i), feature]
        loadDataBase(1)
    end_time = time.time()
    print('run time: ', end_time - start_time)

if __name__ == '__main__':
    face_recognition_test('lfw',500)
    # db_test()
    print('end')


