#!/usr/bin/env python3

import ctypes
import random
from time import time

import torch
import torch.nn as nn
import torchvision.models as models
from progress.bar import IncrementalBar
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import helper

# Placeholder values below
mean = (0.7048001754523248, 0.6353024817539352, 0.5856219251267757)
std = (0.21634347931241812, 0.2423184790247176, 0.2713632622907276)


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
    RandomRotationFromList([0, 90, 270]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-2, 2)),
    prepare
])


def get_training_loader(batch_number):
    training_set = ImageFolder(root=f"../batches/batch-{batch_number}/train",
                               transform=augmented_transforms)

    return DataLoader(
        dataset=training_set,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
        shuffle=True)


def get_validation_loader(batch_number):
    validation_set = ImageFolder(root=f"../batches/batch-{batch_number}/val",
                                 transform=prepare)
    return DataLoader(
        dataset=validation_set,
        batch_size=128,
        num_workers=2,
        shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(type):
    """
    type should be one of (resnext|densenet)
    """
    num_labels = 17

    if type == "resnext":
        model = models.resnext101_32x8d(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_labels),
            nn.LogSoftmax(dim=1))
        return model

    elif type == "densenet":
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
            nn.LogSoftmax(dim=1))
        return model


if __name__ == '__main__':
    for batch_num in range(0, 10):
        model = get_model("resnext")

        # Move to GPU for faster training, if available
        model.to(device)

        error_function = nn.NLLLoss(
            weight=torch.tensor([
                # weights calculated from data_prep function: 100 / number(images_in_class)
                1.00, 9.70, 5.56, 3.16, 7.35, 0.66, 3.97, 2.31, 1.02, 0.41, 0.96, 1.47, 9.09, 1.36, 1.03, 1.83, 1.20])) \
            .to(device)
        optimiser = torch.optim.AdamW(model.parameters(), amsgrad=True)

        max_epochs = 201

        validation_loss_history = []

        training_loader = get_training_loader(batch_number=batch_num)
        validation_loader = get_validation_loader(batch_number=batch_num)

        print(f"Training batch-{batch_num}")

        for epoch in range(max_epochs):

            training_loss, validation_loss, accuracy, counter = 0, 0, 0, 0

            model.train()
            training_bar = IncrementalBar(message='Training  ',
                                          max=len(training_loader),
                                          suffix="%(percent)d%% [%(elapsed_td)s / %(eta_td)s]")
            training_timer = time()

            # -------------------
            #    TRAINING STEP
            # -------------------
            for inputs, labels in training_loader:
                def closure():
                    # Clear the gradients and perform a forward pass
                    optimiser.zero_grad()
                    output = model(inputs)

                    # Check the loss
                    loss = error_function(output, labels)

                    # Clear the gradients and perform a backward pass
                    optimiser.zero_grad()

                    loss.backward()

                    return loss


                inputs, labels = inputs.to(device), labels.to(device)

                # Call the closure and read the loss
                loss = optimiser.step(closure)

                training_loss += loss.item() * inputs.size(0)

                training_bar.next()

            # Training timer
            training_time = helper.with_decimal_places(
                time() - training_timer, 2)
            print(f" | time taken: {training_time} seconds")

            validation_bar = IncrementalBar(message='Validating',
                                            max=len(validation_loader),
                                            suffix="%(percent)d%% [%(elapsed_td)s / %(eta_td)s]")
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

                    validation_loss += loss.item() * inputs.size(0)

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
            validation_time = helper.with_decimal_places(
                time() - validation_timer, 2)
            print(f" | time taken: {validation_time} seconds\n")

            # Calculate and print the losses
            training_loss = training_loss / len(training_loader.dataset)
            validation_loss = validation_loss / len(validation_loader.dataset)

            accuracy = helper.to_percentage(accuracy / len(validation_loader))
            training_loss = helper.with_decimal_places(training_loss, 6)
            validation_loss = helper.with_decimal_places(validation_loss, 6)
            print(f"Epoch {epoch} -> " +
                  f"Accuracy {accuracy} | " +
                  f"Training loss {training_loss} | " +
                  f"Validation loss {validation_loss}")

            validation_loss_history.append(validation_loss)

            # save model when a new record low validation loss has been found
            helper.save_model_if_at_local_minimum(
                model=model,
                file_name=f"../batch-{batch_num}-resnext101-32x8d.pth",
                loss_history=validation_loss_history)

            if not helper.should_continue_training(validation_loss_history):
                print(f"Training stopping early at epoch #{epoch}")
                break

            print("-" * 96)  # Epoch delimiter

        print(f"Batch {batch_num} finished training")
