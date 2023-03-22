# Importing dependencies
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import dlib
import tensorflow as tf


cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Loading the data
meta = pd.read_csv('meta.csv')

# Deleting the age column as we dont need it
meta = meta.drop(['age'], axis=1)

# Spliting the dataset into train and test set
D_train, D_test = train_test_split(meta, test_size=0.1, random_state=42)

# The dataset contains more male faces that female faces. This can couse some problems.
# One feature can start dominating on other feature. To solve this I am selecting equal number of male and female faces in the training set
D_train_male = D_train[D_train['gender'] == 'male']
D_train_female = D_train[D_train['gender'] == 'female']

no_male = len(D_train_male)
no_female = len(D_train_female)

if no_male > no_female:
    extra = D_train_male[no_female:]
    D_train_male = D_train_male[0:no_female]

    D_test = pd.concat((D_test, extra))
else:
    extra = D_train_male[no_male:]
    D_train_male = D_train_male[0:no_male]

    D_test = pd.concat((D_test, extra))

D_train = pd.concat((D_train_male, D_train_female))

# Shuffling the dataset
D_train = D_train.sample(frac=1)
D_test = D_test.sample(frac=1)

# Generating folder struture for the data
output_dir_train_male = 'dataset/gender/train/male'
output_dir_train_female = 'dataset/gender/train/female'

if not os.path.exists(output_dir_train_male):
    os.makedirs(output_dir_train_male)

if not os.path.exists(output_dir_train_female):
    os.makedirs(output_dir_train_female)

output_dir_test_male = 'dataset/gender/test/male'
output_dir_test_female = 'dataset/gender/test/female'

if not os.path.exists(output_dir_test_male):
    os.makedirs(output_dir_test_male)

if not os.path.exists(output_dir_test_female):
    os.makedirs(output_dir_test_female)

# Finally processing the image training and testting set
counter = 0

for image in D_train.values:
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
    if image[0] == 'male':
        cv2.imwrite('dataset/gender/train/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('dataset/gender/train/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1

counter = 0

for image in D_test.values:
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
    if image[0] == 'male':
        cv2.imwrite('dataset/gender/test/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('dataset/gender/test/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1
