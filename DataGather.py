import time

import mediapipe as mp
import cv2
import os
import copy
from PIL import Image
import numpy as np


# CHANGE THESE TO SAVE FILES IN SPECIFIC FOLDERS
directory = "up"  # CHANGE TO GO INTO SPECIFIC FOLDER
fileName = "upleftHand"  # NAME OF FILE (filename)i.jpg is the full name of file when saved where i is the frame count
frameCount = 500  # HOW MANY FRAMES TO SAVE
waitTime = 0.01  # HOW LONG IN SECs TO WAIT

# use this to identify our hands
def findHand(frame):
    # mediapipe uses RGB to find hands, cv2 captures BGR for some reason
    # we correct this
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # use the mediapipe hands model to identify our hand and keypoints
    results = hands.process(img)
    cutFrame = None

    # mediapipe normalizes coordinates, use capture size to fix that for later
    h, w, c = img.shape

    # if a hand is identified...
    if results.multi_hand_landmarks:
        # get the landmarks of the first hand
        hand1 = results.multi_hand_landmarks[0]
        Xs = []
        Ys = []
        # denormalize the coordinates of each landmark
        for landmark in hand1.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            Xs.append(cx)
            Ys.append(cy)

        # identify the bounds of our hand
        xmin = min(Xs)
        xmax = max(Xs)
        ymin = min(Ys)
        ymax = max(Ys)

        cutFrame = copy.deepcopy(frame[max(0, ymin - 10):min(frame.shape[0], ymax + 10), max(0, xmin - 10):min(frame.shape[1], xmax + 10)])

        cv2.rectangle(frame, (max(0, xmin - 10), max(0, ymin - 10)), (min(frame.shape[1], xmax + 10), min(frame.shape[0], ymax + 10)), (0, 255, 0), 2)

    # return the frame, and the cropped frame (if applicable)
    return frame, cutFrame

# define the hands model
hands = mp.solutions.hands.Hands()

# obviously, capture video from the camera
cap = cv2.VideoCapture(0)

# if your camera isn't working, that sucks
if not cap.isOpened():
    print("well this sucks (no camera)")
    exit()

framesCaptured = 0

# keep capturing the camera feed until you quit
while True:
    ret, frame = cap.read()

    if not ret or framesCaptured > frameCount:
        break

    # find hands if available
    frame, cutFrame = findHand(frame)

    # show the frame for reference
    cv2.imshow("frame", frame)

    # if your hand was found and the cropped frame is a valid shape, show it
    if cutFrame is not None:
        if cutFrame.shape[0] != 0 and cutFrame.shape[1] != 0:
            cv2.imshow("cutFrame", cutFrame)
            framesCaptured += 1

            cutFrame = Image.fromarray(cutFrame)
            cutFrame = cutFrame.convert("L")
            cutFrame = cutFrame.resize((128, 128))

            cutFrame = np.array(cutFrame)

            cv2.imwrite("HandData/{}/{}.jpg".format(directory, fileName + str(framesCaptured)), cutFrame)

    # press the 'q' button to quit
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(waitTime)

# release the camera and end the program
cap.release()
cv2.destroyAllWindows()