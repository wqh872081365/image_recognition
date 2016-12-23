# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import os
import MySQLdb


def face(imagePath):
    # Get user supplied values
    # imagePath = 'example/FaceDetect-master/abba.png'
    imageName = 'result_'+imagePath.split('/')[-1]
    cascPath = 'example/haarcascade_frontalface_default.xml'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces in {1}".format(len(faces), imagePath))

    # Draw a rectangle around the faces
    data_list = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        data_list.append([(x, y), (x+w, y+h)])

    cv2.imwrite('test_image_face/'+imageName, image)

    return [imagePath, len(faces), str(data_list), 'test_image_face/'+imageName]


def insert(name, number_detect, number_detect_data, name_detect):
    # 打开数据库连接
    db = MySQLdb.connect("localhost", "root", "password", "face_recognition")

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 插入语句
    sql = "INSERT INTO face(name, \
           number_detect, number_detect_data, name_detect) \
           VALUES ('%s', '%d', '%s', '%s')" % \
          (name, number_detect, number_detect_data, name_detect)
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()

    # 关闭数据库连接
    db.close()


def main():
    file_list = os.listdir('./test_image')
    for file in file_list:
        (name, number_detect, number_detect_data, name_detect) = face('test_image/'+file)
        insert(name, number_detect, number_detect_data, name_detect)


if __name__ == '__main__':
    main()