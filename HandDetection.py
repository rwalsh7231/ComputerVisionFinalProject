import cv2
import mediapipe as mp

def findHand(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    cutFrame = None

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        hand1 = results.multi_hand_landmarks[0]
        Xs = []
        Ys = []
        for landmark in hand1.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            Xs.append(cx)
            Ys.append(cy)

        xmin = min(Xs)
        xmax = max(Xs)
        ymin = min(Ys)
        ymax = max(Ys)

        cutFrame = frame[ymin:ymax, xmin:xmax]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return frame, cutFrame

handsMp = mp.solutions.hands
hands = handsMp.Hands()
mpDraw = mp.solutions.drawing_utils


# obviously, capture video from the camera
cap = cv2.VideoCapture(0)

# if your camera isn't working, that sucks
if not cap.isOpened():
    print("well this sucks (no camera)")
    exit()

# keep capturing the camera feed until you quit
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame, cutFrame = findHand(frame)

    cv2.imshow("frame", frame)

    if cutFrame is not None:
        if cutFrame.shape[0] != 0 and cutFrame.shape[1] != 0:
            cv2.imshow("cutFrame", cutFrame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the camera and end the program
cap.release()
cv2.destroyAllWindows()