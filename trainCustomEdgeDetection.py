from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import copy
import mediapipe as mp

hands = mp.solutions.hands.Hands()
EPOCHS=25

class CustomDataset(Dataset):
    def __init__(self, images, labels):

        images = images.astype(np.float32) / 255.0

        images = images[:, None, :, :]

        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))

        self.finalPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 6))  # 6 classes: 1, 2, 3, 4, 5, 6

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.finalPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Training and testing the model
def train_and_evaluate(X_train, y_train, X_test, y_test):

    train_data = CustomDataset(X_train, y_train) # Creating training dataset
    test_data = CustomDataset(X_test, y_test) # Creating testing dataset

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss() # Using Cross entropy loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    print("Training the model")
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch + 1))
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print("Accuracy: {}".format(acc))

    # Evaluating the model
    print("Evaluating the model")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, model

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

        cutFrame = copy.deepcopy(frame[max(0, ymin-100):min(frame.shape[0], ymax+100), max(0, xmin-100):min(frame.shape[1], xmax+100)])

        cv2.rectangle(frame, (max(0, xmin-100), max(0, ymin-100)), (min(frame.shape[1], xmax+100), min(frame.shape[0], ymax+100)), (0, 255, 0), 2)

    return frame, cutFrame

def main():
    # Load dataset
    print("Loading the dataset")

    #0 = a, 1 = b, 2 = down, 3 = left, 4 = right, 5 = up
    labels = [0, 1, 2, 3, 4, 5]
    directories = ['a', 'b', 'down', 'left', 'right', 'up']

    images = []
    imageLabels = []
    for i in range(len(directories)):
        for fileName in os.listdir(f"HandData/{directories[i]}"):
            # Load the image
            image_path = os.path.join(f"HandData/{directories[i]}", fileName)
            image = cv2.imread(image_path)

            if image is None:
                continue  # skip if image not loaded properly

            # Use findHand to crop the hand
            _, cropped = findHand(image)

            if cropped is None:
                continue  # skip if no hand was detected

            # Resize and convert to grayscale
            cropped = cv2.resize(cropped, (128, 128))
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # === Apply Canny Edge Detection ===
            edge_image = cv2.Canny(cropped, threshold1=100, threshold2=100)

            images.append(edge_image)
            imageLabels.append(labels[i])


    images = np.array(images)
    imageLabels = np.array(imageLabels)

    xTrain, xTest, yTrain, yTest = train_test_split(images, imageLabels, test_size=0.25)

    accuracy, model = train_and_evaluate(xTrain, yTrain, xTest, yTest)

    print("Train success, Accuracy: {}".format(accuracy))

    torch.save(model.state_dict(), 'custom_model_edge_detection.pth')


if __name__ == "__main__":
    main()
