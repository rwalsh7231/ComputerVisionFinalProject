from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

EPOCHS=5

class SignLanguageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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


# Training and testing the model 
def train_and_evaluate(params, X_train, y_train, X_test, y_test):
    batch_size = params['batch_size']
    lr = params['learning_rate']
    optimizer_type = params['optimizer']

    train_data = SignLanguageDataset(X_train, y_train) # Creating training dataset
    test_data = SignLanguageDataset(X_test, y_test) # Creating testing dataset

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss() # Using Cross entropy loss

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("Training the model")
    for epoch in range(EPOCHS):
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
    train_df = pd.read_csv('ComputerVisionFinalProject/data/sign_mnist_train.csv')
    test_df = pd.read_csv('ComputerVisionFinalProject/data/sign_mnist_test.csv')

    X_train = train_df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    y_test = test_df.iloc[:, 0].values

    # Define hyperparameter grid
    print("Defining the hyperparameter grid")
    param_grid = {
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.0005],
        'optimizer': ['adam', 'sgd']
    }

    best_acc = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"🔍 Trying params: {params}")
        acc, model = train_and_evaluate(params, X_train, y_train, X_test, y_test)
        print(f"→ Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    # Save best model
    if best_model:
        torch.save(best_model.state_dict(), 'best_model.pth')
        print("\n Best model saved as 'best_model.pth'")

    print("\n🏆 Best Hyperparameters:", best_params)
    print(f"🏆 Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
