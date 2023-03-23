from flask import Flask, render_template, Response
import tensorflow
import cv2
from OpenSSL import SSL
from tensorflow.keras import applications
import numpy as np
from PIL import Image
# import tensorflow as tf
from tensorflow.keras.models import load_model
import random
from tensorflow.keras.utils import get_file
from omegaconf import OmegaConf
# import dlib
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as Ke
from pathlib import Path
import cv2
import h5py
import numpy as np



def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def get_model(cfg):
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces



  
camera = cv2.VideoCapture(0)
def generate_frames():



    pretrained_model = "models/age_gender_merged.hdf5"
    weight_file = get_file("age_gender_merged.hdf5", pretrained_model, cache_subdir="models", cache_dir=str(Path(__file__).resolve().parent))



        # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    
    while True:
        # Read frame from webcam stream
        ret, img = camera.read()

        if not ret:
            break
        else:

            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            detected = detect_faces(input_img)
            faces = np.empty((len(detected), img_size, img_size, 3))

            margin = 0.4
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x, y, w, h = d
                    x1, y1, x2, y2, w, h = x, y, x + w + 1, y + h + 1, w, h
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                results = model.predict(faces)

                predicted_genders = results[0]
                
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                

                for i, d in enumerate(detected):
                    label = "{}, {}".format(int(predicted_ages[i]),
                                            "M" if predicted_genders[i][0] < 0.5 else "F")
                    draw_label(img, (x1, y1), label)

                cv2.imshow("result", img)
                key = cv2.waitKey(30)

                if key == 27:  # ESC
                    break

    camera.release()
    cv2.destroyAllWindows()
    

generate_frames()
