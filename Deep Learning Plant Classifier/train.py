#!/usr/bin/env python3

import ctypes
from progress.bar import FillingSquaresBar, FillingCirclesBar
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from time import time
import torchvision.models as models
import torch.nn as nn
import torch
import random
import helper

# Windows workaround for LoadLibraryA issue
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# Placeholder values below
# TODO: replace them with actual mean and std, by computing them over the dataset
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)


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

# dataloaders
training_loader = DataLoader(
    dataset=training_set,
    batch_size=128,
    num_workers=1,
    pin_memory=True,
    drop_last=True,  # TODO: test the effects of this on convergence rate
    shuffle=True)

validation_loader = DataLoader(
    dataset=validation_set,
    batch_size=128,
    num_workers=1,
    shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # load pretrained densenet-161 model
    model = models.densenet161(pretrained=True)

    # turn training off for all parameters first
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier_input = model.classifier.in_features
    num_labels = 17

    # replace the classifier layer
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, 512),
                               nn.ReLU(),
                               nn.Linear(512, num_labels),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    # Move to GPU for faster training, if available
    model.to(device)

    error_function = nn.NLLLoss()
    optimiser = torch.optim.AdamW(classifier.parameters())

    validation_loss_history = []

    max_epochs = 201
    for epoch in range(max_epochs):
        training_loss, validation_loss, accuracy, counter = 0, 0, 0, 0

        model.train()
        training_bar = FillingCirclesBar(message='Training  ',  # extra space to align with validation bar
                                         max=len(training_loader))
        training_timer = time()

        # -------------------
        #    TRAINING STEP
        # -------------------
        for inputs, labels in training_loader:
            def closure():
                # Clear the gradients
                optimiser.zero_grad()

                output = model(inputs)

                # Check the loss
                loss = error_function(output, labels)
                loss.backward()

                return loss

            inputs, labels = inputs.to(device), labels.to(device)

            loss = optimiser.step(closure)

            training_loss += loss.item()*inputs.size(0)

            training_bar.next()

        # Training timer
        print(
            f" | time taken: {helper.decimal_places(format(time() - training_timer), 2)} seconds")

        validation_bar = FillingSquaresBar(message='Validating',
                                           max=len(validation_loader))
        validation_timer = time()

        # -------------------
        #   VALIDATION STEP
        # -------------------
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass, calculate validation loss
                output = model.forward(inputs)

                loss = error_function(output, labels)

                validation_loss += loss.item()*inputs.size(0)

                # # Reverse the log part of LogSoftMax
                output = torch.exp(output)

                # Get top class of outputs, tested for top-1
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # Calculate mean, add it to running accuracy for current epoch
                accuracy += torch.mean(
                    equals.type(torch.FloatTensor)).item()

                validation_bar.next()

        # Validation timer
        print(
            f" | time taken: {helper.decimal_places(time() - validation_timer, 2)} seconds")

        # Calculate and print the losses
        training_loss = training_loss / len(training_loader.dataset)
        validation_loss = validation_loss / len(validation_loader.dataset)

        print(f'''\nEpoch {epoch} recap
        Accuracy:        {helper.to_percentage(accuracy/len(validation_loader))}
        Training loss:   {helper.decimal_places(training_loss, 6)}
        Validation loss: {helper.decimal_places(validation_loss, 6)}''')

        validation_loss_history.append(validation_loss)

        # save model when a new record low validation loss has been found
        helper.save_model_if_at_local_minimum(
            model=model,
            file_name="../densenet-161.pth",
            loss_history=validation_loss_history)

        if not helper.should_continue_training(validation_loss_history):
            print(
                f"Training stopping early at epoch #{epoch}")
            break

        print("-"*96)  # Epoch delimiter
