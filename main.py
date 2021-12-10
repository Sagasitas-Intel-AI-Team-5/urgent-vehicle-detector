# https://github.com/GauravSahani1417/OpenCV-Implementaion

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ctypes  # An included library with Python install.
from keras.models import load_model
import cv2
import numpy as np
import playsound

done = False
model_file = 'model.h5'
size = 180
# here is the animation

model = load_model(model_file)
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')


def alert(text):
    return ctypes.windll.user32.MessageBoxW(0, text, "alert...!", 1)


def get_label(img):
    img = cv2.resize(img, (size, size))
    img = np.reshape(img, [1, size, size, 3])
    class_name = ['ambulance', 'damkar', 'mobil', 'motor', 'sepeda']

    # classes = loaded_model.predict(img)
    classes = model.predict(img)
    class_index = np.argmax(classes)
    x = class_name[class_index]
    return(x)


cap = cv2.VideoCapture('traffic.mp4')
while cap.isOpened():
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in cars:
     #    car = frame.crop(x, y, x+w, y+h)
        cropped_image = frame[y:y+h, x:x+w]
        clas = get_label(cropped_image)
        if clas == "sepeda" or clas == "motor" or clas == "ambulance" or clas == "damkar":
            clas = "mobil"
        # if clas == "ambulance" or clas == 'damkar':
            # alert("Perhatian...! ada "+clas +", silahkan menepi untuk memberi jalan...!")
            # playsound.playsound('alert.mp3')

        cv2.putText(frame, clas, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    if cropped_image.any():
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('window', frame)
        cv2.setWindowProperty('window', cv2.WND_PROP_TOPMOST, 1)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()