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


testing_set = ImageFolder(root="../batches/batch-0/test",
                          transform=prepare)

testing_loader = DataLoader(dataset=testing_set,
                            batch_size=32,
                            shuffle=False)


# Tests the model
def test_model(model_to_test):
    accuracy = 0
    testing_bar = FillingSquaresBar(message='Testing',
                                    max=len(testing_loader))

    for inputs, labels in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # predict inputs, and reverse the LogSoftMax
        real_predictions = torch.exp(model_to_test(inputs))

        # Get top class of outputs, tested for top-1
        top_p, top_class = real_predictions.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        # Calculate mean, add it to running accuracy for current testing batch
        accuracy += torch.mean(
            equals.type(torch.FloatTensor)).item()

        testing_bar.next()

    testing_accuracy = accuracy / len(testing_loader)
    print(f"\nAccuracy -> {'{:.4f}'.format(testing_accuracy * 100)}%")


if __name__ == '__main__':
    # Evaluate all stored network states
    # -10 is the network state pre-training, for a baseline
    # 0 is after the first epoch, and so on till 190
    saved_model_name = f"../densenet-161.pth"
    print(f"Evaluating model {saved_model_name}")
    saved_model = torch.load(saved_model_name)
    saved_model.eval()

    test_model(model_to_test=saved_model)
