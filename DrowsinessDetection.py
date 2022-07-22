import imutils
from imutils.video import VideoStream
from imutils import face_utils

import cv2
import dlib  # https://miai.vn/2020/03/12/tips-cach-cai-dlib-tren-window-nhanh-gon-khong-bi-loi/

import numpy as np

from pygame import mixer

import time

def euclidean_distance(pointA, pointB):
    return np.linalg.norm(pointA - pointB)


def eye_aspect_ratio(eye):
    eye_width = euclidean_distance(eye[0], eye[3])
    eye_height_1 = euclidean_distance(eye[1], eye[5])
    eye_height_2 = euclidean_distance(eye[2], eye[4])

    EAR = (eye_height_1 + eye_height_2) / (2.0 * eye_width)

    return EAR


closed_EAR = 0.3
score = 10

alarm_on = False

mixer.init()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(1.0)  # Wait 1 second for camera warm-up

while True:
    frame = imutils.resize(vs.read(), width=450)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rectangles = detector.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x, y, w, h) in rectangles:
        rectangle = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = face_utils.shape_to_np(predictor(gray_frame, rectangle))

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        # Draw eye recognition frame
        cv2.drawContours(frame, [left_eye_hull], -1, (150, 75, 38), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (150, 75, 38), 1)

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        EAR = (leftEAR + rightEAR) / 2.0

        if EAR < closed_EAR:
            score -= 1
            if score <= 0:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (47, 33, 191), 2)
                if not alarm_on:
                    mixer.music.load('DayDiBanOi-TongMinh.mp3')
                    mixer.music.play()
                    alarm_on = True

        if EAR > closed_EAR:
            score = 10
            mixer.music.stop()
            alarm_on = False

        cv2.putText(frame, "EAR: {:.3f}".format(EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (118, 179, 39), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
