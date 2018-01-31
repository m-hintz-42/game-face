import numpy as np
import cv2
import time
from keras.models import load_model


def detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 2, 6)

    image_with_detections = np.copy(image)
    facial_keypoints = []

    for (x, y, w, h) in faces:
        face_img = image_with_detections[y:y + h, x:x + w, :]

        # Pre-process image to 96x96 gray
        face_resized = cv2.resize(face_img, (96, 96))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        gray_norm = (face_gray / 255.)[None, :, :, None]

        # Use Model to predict facial keypoint locations
        model = load_model('../my_model.h5')
        keypoints = model.predict(gray_norm)
        keypoints = (keypoints * 48) + 48

        # Normalize image poisitions back to original
        x_pos = keypoints[0][0::2]
        x_norm = x_pos * w / 96 + x

        y_pos = keypoints[0][1::2]
        y_norm = y_pos * h / 96 + y

        facial_keypoints.append((x_norm, y_norm))

        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for face in facial_keypoints:
        for x, y in zip(face[0], face[1]):
            cv2.circle(image_with_detections, (x, y), 3, (0, 255, 0), -1)

    return image_with_detections


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

        image = detector(frame)

        cv2.imshow("face detection activated", image)

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
        time.sleep(0.01)  # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()

laptop_camera_go()