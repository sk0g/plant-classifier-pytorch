#!/usr/bin/env python3

import torch
from matplotlib import pyplot
from progress.bar import FillingSquaresBar
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from train import prepare, device, helper

testing_set = ImageFolder(root="../batches/batch-0/test",
                          transform=prepare)

testing_loader = DataLoader(dataset=testing_set,
                            batch_size=128,
                            shuffle=False)

class_names = [
    "background",
    "bertya calycina",
    "bertya ernestiana",
    "bertya glandulosa",
    "bertya granitica",
    "bertya pedicellata",
    "bertya pinifolia",
    "bertya recurvata",
    "bertya sharpeana",
    "grevillea glossadenia",
    "grevillea hockingsii",
    "grevillea hodgei",
    "grevillea kennedyana",
    "grevillea linsmithii",
    "grevillea quadricauda",
    "grevillea scortechinii",
    "grevillea venusta"
]


def display_confusion_matrix(truth_list, predictions_list):
    conf_matrix = confusion_matrix(
        y_true=truth_list,
        y_pred=predictions_list,
        normalize='all')

    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names) \
        .plot(xticks_rotation=90, cmap=pyplot.cm.Blues)

    pyplot.show()


# Tests the model
def test_model(model_to_test):
    model_to_test.eval()

    truth_list, predictions_list = [], []
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

        # append to confusion matrix lists
        for truth, prediction in zip(labels.view(-1), top_1_class.view(-1)):
            predictions_list.append(prediction.item())
            truth_list.append(truth.item())

        testing_bar.next()

    top_1_testing_accuracy = top_1_accuracy / len(testing_loader.dataset)
    top_5_testing_accuracy = top_5_accuracy / len(testing_loader.dataset)
    print(f'''\nAccuracy
        top-1: {helper.to_percentage(top_1_testing_accuracy)}
        top-5: {helper.to_percentage(top_5_testing_accuracy)}''')

    print("Displaying confusion matrix...")
    display_confusion_matrix(truth_list, predictions_list)


if __name__ == '__main__':
    # Evaluate all stored network states
    # -10 is the network state pre-training, for a baseline
    # 0 is after the first epoch, and so on till 190
    saved_model_name = f"../densenet-161.pth"
    print(f"Evaluating model {saved_model_name}")
    saved_model = torch.load(saved_model_name)
    saved_model.eval()

    test_model(model_to_test=saved_model)
