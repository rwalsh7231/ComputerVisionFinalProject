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

EPOCHS=10

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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.finalPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 6)  # 6 classes: 1, 2, 3, 4, 5, 6

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.finalPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Training and testing the model
def train_and_evaluate(X_train, y_train, X_test, y_test):

    train_data = CustomDataset(X_train, y_train) # Creating training dataset
    test_data = CustomDataset(X_test, y_test) # Creating testing dataset

    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss() # Using Cross entropy loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("Training the model")
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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


def main():
    # Load dataset
    print("Loading the dataset")

    labels = [0, 1, 2, 3, 4, 5]
    directories = ['a', 'b', 'down', 'left', 'right', 'up']

    images = []
    imageLabels = []
    for i in range(len(directories)):
        for fileName in os.listdir("HandData/{}".format(directories[i])):
            image = Image.open(os.path.join("HandData/{}".format(directories[i]), fileName))
            image = image.resize((250, 250))
            image = image.convert('L')
            image = np.array(image)
            images.append(image)
            imageLabels.append(labels[i])


    images = np.array(images)
    imageLabels = np.array(imageLabels)

    xTrain, xTest, yTrain, yTest = train_test_split(images, imageLabels, test_size=0.1)

    accuracy, model = train_and_evaluate(xTrain, yTrain, xTest, yTest)

    print("Train success, Accuracy: {}".format(accuracy))

    torch.save(model.state_dict(), 'custom_model.pth')




if __name__ == "__main__":
    main()
