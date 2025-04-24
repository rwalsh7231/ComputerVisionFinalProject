import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the CNN model as in train_model.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 25)  # 25 classes (A-Y, no J)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the transformation to convert the cropped frame into the format the model expects
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the best model
model = CNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Initialize the mediapipe hands model
hands = mp.solutions.hands.Hands()

# Function to find the hand and crop the frame
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

        cutFrame = frame[max(0, ymin-20):min(frame.shape[0], ymax+20), max(0, xmin-20):min(frame.shape[1], xmax+20)]

        cv2.rectangle(frame, (max(0, xmin-20), max(0, ymin-20)), (min(frame.shape[1], xmax+20), min(frame.shape[0], ymax+20)), (0, 255, 0), 2)

    return frame, cutFrame

# Capture video
cap = cv2.VideoCapture(0)

# Check if the camera is available
if not cap.isOpened():
    print("Unable to access camera")
    exit()

# Function to predict using the model
def predict_hand(cut_frame):
    pil_image = Image.fromarray(cut_frame)
    image = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the class with the highest probability

    return predicted.item()  # Return the predicted class

# Process the video frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find the hand and get the cropped frame
    frame, cutFrame = findHand(frame)

    if cutFrame is not None:
        if cutFrame.shape[0] != 0 and cutFrame.shape[1] != 0:
            # Display the cropped frame
            cv2.imshow("cutFrame", cutFrame)

            # Predict the class of the hand gesture
            predicted_class = predict_hand(cutFrame)
            print(f"Predicted Class: {predicted_class}")

    # Display the frame
    cv2.imshow("frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
