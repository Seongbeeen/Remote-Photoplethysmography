import cv2
import os


def face_rect(vids, every_frame=False):
    face_rect = []
    dpath = "../src/haarcascade_frontalface_alt.xml"
    if not os.path.exists(dpath):
        print("Cascade file not present!")
    face_cascade = cv2.CascadeClassifier(dpath)
    if not every_frame:
        for image in vids:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = list(face_cascade.detectMultiScale(gray, 1.1, 4))
            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])
                face_rect = detected[-1]
                break
    else:
        image = vids
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = list(face_cascade.detectMultiScale(gray, 1.1, 4))
        detected.sort(key=lambda a: a[-1] * a[-2])
        face_rect = detected[-1]
    return face_rect
