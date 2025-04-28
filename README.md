# ComputerVisionFinalProject

## Created by Robert Walsh, Jash Narwani, and Haris Naveed

### Github Link: https://github.com/rwalsh7231/ComputerVisionFinalProject

This project is a combination of two methods. A prebuilt object detection model for identifying hands from the mediapipe package, and a Convolutional Neural Network for Gesture Identification.
The purpose of these methods is to combine the two and make a quick and efficient gesture recognition system. In our case, we have applied our system to act as a controller replacement for games such as with the NES.

To see what we have accomplished, access the Emulated Controller notebook in this directory. Here you will see how the system works and examples of its use case with example data.
We also recommend looking at the DataGather and trainCustom files to see how we gathered our dataset in the HandData folder and trained our model.

We have also experimented with other pre-built models, but they are not used in our final version, here is one such model: https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main

We have also recorded our experiences with our models, that way one can still see the capabilities of the models even under less than ideal circumstances

## Details of Files

### Emulated Controller.ipynb

This is the main file for our system, it is here where you can experience and use the systems we created.
It supports activities for both using the model as intended or just demonstrating its functionality with an example dataset

### Data Gather.py

This is the file we used to generate our main dataset in the HandData folder. Simply put, the program is designed to take a series of photos of an individual's hands (aided by the mediapipe package).
These images are then convered to grayscale, resized to 128x128 images and stored into their respective folder for future training.
For data gathering, we gathered varied data with this method, 1000 images per specific hand gesture with many variants such as hand motion, distance, and lighting accounted for.

### trainCustom.py

This is the file we used to train the custom_model.pth model. We define our model with three layers of convolutions each put through batch norms and max pooling. The model then goes through one last adaptive pooling and outputs 6 possible values for each of our hand gestures.
To help with accuracy, we applied transforms to the image to make it more difficult for the model to predict, thus improving its overall robustness.
We checked its accuracy each epoch, and watched as it slowly became better.

### HandDetection.py

This is an early version, and base, of our hand detection script. Using the mediapipe Hands() model, we were able to identify and crop frames to just show our hands in an image. It was our hope that by doing this, we can make a more accurate gesture recognition model

## hand-gesture-recognition-mediapipe-main Directory

This is the other model we developed by modifiying the git source code from this repo: https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main. In the folder there is an app.py which exectures the main script and starts the hand modeling gesture recognizer. It uses the model developed by keypoint_classification_EN.ipynb trough the file keypoint_classification.ipynb. We modified it by adding classes and adjusting the existing model to accomidate for those classes. We also added functionality to press the buttons needed for the NES retroarch emulator. This was done in order to control mario in the NES emulator. There is a demo of this model in the recordings directory called CompVdemo.txt which leads to a youtube video of the demo

## AI Usage

We have indeed used ChatGPT to help us with improving the accuracy of our model. When we first began training our model, the best we could get was about 40% even after many epochs.
With the usage of the AI, we were able to determine a better setup for our model's layers. We converted our layers to

### self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))

This is in comparison of our layers originally being:

### self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

### Old Versions Folder

In the trainCustomEdgeDetection.py, we used AI to add some edge detection and how to properly implement that. For example, this code (edge_image = cv2.Canny(image, threshold1=100, threshold2=100)) came from AI. In the model architectures, AI gave some suggestions to improve the model. For example, trying different optimizers like Adam or SGD. Whenever there was an error that did not make sense, we use AI to understand what this error is about and how to resolve it.
