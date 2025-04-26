import torchvision.transforms
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

transform = transforms.Compose([
    torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3)
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform):

        images = images.astype(np.float32) / 255.0

        images = images[:, None, :, :]

        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = self.transform(image)

        return image, label


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
def train_and_evaluate(X_train, y_train, X_test, y_test, model=None):

    train_data = CustomDataset(X_train, y_train, transform) # Creating training dataset
    test_data = CustomDataset(X_test, y_test, transform) # Creating testing dataset

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    if not model:
        model = CNN()

    criterion = nn.CrossEntropyLoss() # Using Cross entropy loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

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


def main():

    if os.path.exists("custom_model.pth"):
        model = CNN()
        model.load_state_dict(torch.load("custom_model.pth"))
    else:
        model = None

    # Load dataset
    print("Loading the dataset")

    #0 = a, 1 = b, 2 = down, 3 = left, 4 = right, 5 = up
    labels = [0, 1, 2, 3, 4, 5]
    directories = ['a', 'b', 'down', 'left', 'right', 'up']

    images = []
    imageLabels = []
    for i in range(len(directories)):
        for fileName in os.listdir("HandData/{}".format(directories[i])):
            image = Image.open(os.path.join("HandData/{}".format(directories[i]), fileName))
            image = np.array(image)
            images.append(image)
            imageLabels.append(labels[i])


    images = np.array(images)
    imageLabels = np.array(imageLabels)

    xTrain, xTest, yTrain, yTest = train_test_split(images, imageLabels, test_size=0.15)

    accuracy, model = train_and_evaluate(xTrain, yTrain, xTest, yTest, model)

    print("Train success, Accuracy: {}".format(accuracy))

    torch.save(model.state_dict(), 'custom_model.pth')




if __name__ == "__main__":
    main()
