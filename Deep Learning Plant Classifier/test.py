#!/usr/bin/env python3

from torchvision import transforms
from progress.bar import FillingSquaresBar
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train import prepare, device
import torchvision.models as models
import torch.nn as nn
import torch
import random
import helper


testing_set = ImageFolder(root="../batches/batch-0/test",
                          transform=prepare)

testing_loader = DataLoader(dataset=testing_set,
                            batch_size=128,
                            shuffle=False)


# Tests the model
def test_model(model_to_test):
    model_to_test.eval()

    top_1_accuracy, top_5_accuracy = 0, 0
    testing_bar = FillingSquaresBar(message='Testing',
                                    max=len(testing_loader))

    for inputs, labels in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # predict inputs, and reverse the LogSoftMax
        real_predictions = torch.exp(model_to_test(inputs))

        # Get top class of outputs
        _, top_1_class = real_predictions.topk(k=1)
        _, top_5_classes = real_predictions.topk(k=5)

        # Run predictions
        top_1_equals = top_1_class == labels.view(*top_1_class.shape)
        top_5_equals = top_5_classes == labels.view(*top_1_class.shape)

        # Count all the accurate guesses
        top_1_accuracy += top_1_equals.sum().item()
        top_5_accuracy += top_5_equals.sum().item()

        testing_bar.next()

    top_1_testing_accuracy = top_1_accuracy / len(testing_loader.dataset)
    top_5_testing_accuracy = top_5_accuracy / len(testing_loader.dataset)
    print(f'''\nAccuracy
        top-1: {helper.to_percentage(top_1_testing_accuracy)}
        top-5: {helper.to_percentage(top_5_testing_accuracy)}''')


if __name__ == '__main__':
    # Evaluate all stored network states
    # -10 is the network state pre-training, for a baseline
    # 0 is after the first epoch, and so on till 190
    saved_model_name = f"../densenet-161.pth"
    print(f"Evaluating model {saved_model_name}")
    saved_model = torch.load(saved_model_name)
    saved_model.eval()

    test_model(model_to_test=saved_model)
