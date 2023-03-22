# Importing dependencies
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import dlib
import tensorflow as tf


cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")


# Loading dataset
meta = pd.read_csv('meta.csv')

# Dropping gender column
meta = meta.drop(['gender'], axis=1)

# Filtaring dataset
meta = meta[meta['age'] >= 0]
meta = meta[meta['age'] <= 101]

# Converting into numpy array
meta = meta.values

# Spliting dataset into training and testing set
D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)

# Making the directory structure
for i in range(102):
    output_dir_train_male = 'dataset/age/train/' + str(i)
    output_dir_train_female = 'dataset/age/train/' + str(i)

    if not os.path.exists(output_dir_train_male):
        os.makedirs(output_dir_train_male)

    if not os.path.exists(output_dir_train_female):
        os.makedirs(output_dir_train_female)

    output_dir_test_male = 'dataset/age/test/' + str(i)
    output_dir_test_female = 'dataset/age/test/' + str(i)

    if not os.path.exists(output_dir_test_male):
        os.makedirs(output_dir_test_male)

    if not os.path.exists(output_dir_test_female):
        os.makedirs(output_dir_test_female)

# Finally making the training and testing set
counter = 0

for image in D_train:
    img = cv2.imread(image[1], 1)
    faces_cnn = cnn_face_detector(img, 1)
    if len(faces_cnn) == 1:
        offset_x , offset_y  = max(faces_cnn[0].rect.left(),0),max(faces_cnn[0].rect.top(),0)
        target_width, target_height = faces_cnn[0].rect.right() - offset_x, faces_cnn[0].rect.bottom() - offset_y

        target_width = min(target_width, img.shape[1]-offset_x)
        target_height = min(target_height, img.shape[0]-offset_y)

        face_img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, target_height,target_width)
        img = face_img
        img = img.numpy()
    img = cv2.resize(img, (128,128))
    cv2.imwrite('dataset/age/train/' + str(image[0]) + '/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1

counter = 0

for image in D_test:
    img = cv2.imread(image[1], 1)
    faces_cnn = cnn_face_detector(img, 1)
    if len(faces_cnn) == 1:
        offset_x , offset_y  = max(faces_cnn[0].rect.left(),0),max(faces_cnn[0].rect.top(),0)
        target_width, target_height = faces_cnn[0].rect.right() - offset_x, faces_cnn[0].rect.bottom() - offset_y

        target_width = min(target_width, img.shape[1]-offset_x)
        target_height = min(target_height, img.shape[0]-offset_y)

        face_img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, target_height,target_width)
        img = face_img
        img = img.numpy()
    img = cv2.resize(img, (128,128))
    cv2.imwrite('dataset/age/test/' + str(image[0]) +  '/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1

