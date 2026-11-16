# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
<img width="1036" height="717" alt="Screenshot 2026-11-16 180957" src="https://github.com/user-attachments/assets/a173a903-a71f-4915-9fff-80758010ff93" />


## DESIGN STEPS
### STEP 1: 
Load the Fashion-MNIST dataset and preprocess images using tensor conversion and normalization.

###STEP 2:
Create DataLoaders for training and testing with a fixed batch size.

###STEP 3:
Design a CNN model with convolution, ReLU activation, max-pooling, and fully connected layers.

###STEP 4:
Train the model using Cross-Entropy loss and the Adam optimizer for several epochs.

###STEP 5:
Test the trained model to calculate accuracy, confusion matrix, and classification report.

###STEP 6:
Predict and display the class of a single test image with actual and predicted labels.

## PROGRAM

### Name: S.Nithyasri

### Register Number: 212225040289

```
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



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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


```

### OUTPUT

## Training Loss per Epoch

<img width="532" height="307" alt="Screenshot 2026-11-16 181408" src="https://github.com/user-attachments/assets/2f9555ef-788f-4213-b129-281450713424" />


## Confusion Matrix


<img width="692" height="590" alt="Screenshot 2026-11-16 181440" src="https://github.com/user-attachments/assets/2a164716-f730-4319-8cbc-c6f148e2cb8a" />

## Classification Report

<img width="525" height="380" alt="Screenshot 2026-11-16 181514" src="https://github.com/user-attachments/assets/6ffefd6d-6ac1-4d9e-bb30-ecb156d25a4e" />

### New Sample Data Prediction
<img width="496" height="570" alt="Screenshot 2026-11-16 181606" src="https://github.com/user-attachments/assets/9f4e40ea-edeb-4d61-8f82-e1bf7a847c90" />


## RESULT
this program running successfully
