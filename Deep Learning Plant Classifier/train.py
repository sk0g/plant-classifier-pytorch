#!/usr/bin/env python3

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch
import random

from progress.bar import FillingSquaresBar, FillingCirclesBar
# Windows workaround for LoadLibraryA issue
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]


class RandomRotationFromList:
    # 90 degree rotating transform
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


# transforms
prepare = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

augmented_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    RandomRotationFromList([0, 90, 180, 270]),
    prepare
])

# datasets
training_set = ImageFolder(root="../batches/batch-0/train",
                           transform=augmented_transforms)

validation_set = ImageFolder(root="../batches/batch-0/val",
                             transform=prepare)

testing_set = ImageFolder(root="../batches/batch-0/test",
                          transform=prepare)

# dataloaders
training_loader = DataLoader(dataset=training_set,
                             batch_size=32,
                             shuffle=True)

validation_loader = DataLoader(dataset=validation_set,
                               batch_size=32,
                               shuffle=True)

testing_loader = DataLoader(dataset=testing_set,
                            batch_size=32,
                            shuffle=True)


# load pretrained densenet-161 model
model = models.densenet161(pretrained=True)

# turn training off for all parameters first
for parameter in model.parameters():
    parameter.requires_grad = False

classifier_input = model.classifier.in_features
num_labels = 3

# replace the classifier layer
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

# Move to GPU for faster training, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

error_function = nn.NLLLoss()
optimiser = torch.optim.AdamW(classifier.parameters())

epochs = 100
for epoch in range(epochs):
    training_loss, validation_loss, accuracy, counter = 0, 0, 0, 0

    model.train()
    training_bar = FillingCirclesBar(message='Training',
                                     max=len(training_loader))
    for inputs, labels in training_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear the gradients
        optimiser.zero_grad()

        output = model.forward(inputs)

        # Check the loss
        loss = error_function(output, labels)
        loss.backward()

        optimiser.step()

        training_loss += loss.item()*inputs.size(0)

        training_bar.next()

    validation_bar = FillingSquaresBar(message='Validating',
                                       max=len(validation_loader))
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass, calculate validation loss
            output = model.forward(inputs)
            validation_loss = error_function(output, labels)
            validation_loss += validation_loss.item()*inputs.size(0)

            # Reverse the log part of LogSoftMax
            real_output = torch.exp(output)

            # Get top class of outputs, tested for top-1
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # Calculate mean, add it to running accuracy for current epoch
            accuracy += torch.mean(
                equals.type(torch.FloatTensor)).item()

            validation_bar.next()

    # Calculate and print the losses
    training_loss = training_loss / len(training_loader.dataset)
    validation_loss = validation_loss / len(validation_loader.dataset)

    print(f"Accuracy -> {accuracy/len(validation_loader)}")
    print(f"Epoch {epoch} recap -> \t|\t Training loss: {'{:.4f}'.format(training_loss)} \t|\t Validation loss: {'{:.4f}'.format(validation_loss)}")
