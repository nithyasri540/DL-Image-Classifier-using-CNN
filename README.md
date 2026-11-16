# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
<img width="867" height="877" alt="Screenshot 2026-11-16 175536" src="https://github.com/user-attachments/assets/2a8f5e53-826e-4254-9d3e-813737a925a0" />


## DESIGN STEPS
### STEP 1: 
Load the Fashion-MNIST dataset and preprocess images using tensor conversion and normalization.

### STEP 2: 

Create DataLoaders for training and testing with a fixed batch size.


### STEP 3: 

Design a CNN model with convolution, ReLU activation, max-pooling, and fully connected layers.

### STEP 4: 
Train the model using Cross-Entropy loss and the Adam optimizer for several epochs.


### STEP 5: 

Test the trained model to calculate accuracy, confusion matrix, and classification report.

### STEP 6: 
Predict and display the class of a single test image with actual and predicted labels.




## PROGRAM

### Name: S.Nithyasri

### Register Number: 212225040289

```

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Number of training samples:", len(train_dataset))
image, label = test_dataset[0]
print("Image shape:", image.shape)
print("Number of testing samples:", len(test_dataset))
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:S.Nithyasri      ')
print('Register Number:212225040289       ')
summary(model, input_size=(1, 28, 28))
# Initialize model, loss function, and optimizer
model =  CNNClassifier().to(device)
criterion =nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name:  S.Nithyasri      ')
        print('Register Number:212225040289       ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model, train_loader, num_epochs=10)
## Step 4: Test the Model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: ')
    print('Register Number: ')
    print(f'Test Accuracy: {accuracy:.4f}')
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: ')
    print('Register Number: ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    # Print classification report
    print('Name: ')
    print('Register Number: ')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
test_model(model, test_loader)
## Step 5: Predict on a Single Image
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    if torch.cuda.is_available():
        image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    class_names = [str(i) for i in range(10)]

    print('Name:S.nithyasri ')
    print('Register Number: 212225040289')
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
predict_image(model, image_index=80, dataset=test_dataset)
```

### OUTPUT

## Training Loss per Epoch

<img width="532" height="307" alt="Screenshot 2026-11-16 181408" src="https://github.com/user-attachments/assets/bb023c07-ff9f-432a-9f77-ec22c5fdbb3c" />



## Confusion Matrix


<img width="692" height="590" alt="Screenshot 2026-11-16 181440" src="https://github.com/user-attachments/assets/1648e2bb-c375-4669-87a8-7799f9cc72a8" />


## Classification Report

<img width="525" height="380" alt="Screenshot 2026-11-16 181514" src="https://github.com/user-attachments/assets/ea74f653-a391-4f53-a099-8703ac02662e" />


### New Sample Data Prediction


<img width="496" height="570" alt="Screenshot 2026-11-16 181606" src="https://github.com/user-attachments/assets/b4c3a1c2-1bee-4b6e-acf5-5b1cab955f3e" />

## RESULT
This program has been executed successfully.
