import numpy as np
# import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
# from PIL import Image
import time

import time


# wrapper function for face/eye detection with your laptop camera
def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # Keep the video stream open
    while rval:
        rval, frame = vc.read()
        image_with_detections = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Face
        face_cascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 2, 6)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Eyes
        eye_cascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 2, 6)
        for (x, y, w, h) in eyes:
            cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("face detection activated", image_with_detections)

        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0:  # Exit by pressing any key
            # Destroy windows
            cv2.destroyAllWindows()

            # Make sure window closes on OSx
            for i in range(1, 5):
                cv2.waitKey(1)
            return

        # Read next frame
        time.sleep(0.05)  # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()

# Call the laptop camera face/eye detector function above
laptop_camera_go()