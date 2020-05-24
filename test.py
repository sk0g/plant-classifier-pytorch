#!/usr/bin/env python3
import os
import sys

import numpy as np
import torch
from progress.bar import FillingSquaresBar
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import confusionMatrixPrettyPrint
from train import prepare, device, helper


def get_testing_loader(batch_num):
    testing_set = ImageFolder(root=f"../batches/batch-{batch_num}/test",
                              transform=prepare)

    return DataLoader(dataset=testing_set,
                      batch_size=256,
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


def print_per_class_accuracy(truth_list, predictions_list):
    tests = [[] for x in range(len(class_names))]

    for truth, prediction in zip(truth_list, predictions_list):
        is_correct = truth == prediction

        tests[truth].append(is_correct)

    for index, test_results in enumerate(tests):
        accuracy = sum(test_results) / len(test_results)
        line_to_print = f"Accuracy for {class_names[index]}".ljust(
            40) + f"{helper.to_percentage(accuracy)}".rjust(10)
        print(f"\t\t{line_to_print}")


# Tests the model
def test_model(model_to_test):
    testing_loader = get_testing_loader(batch_num=0)

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

    print("Calculating and printing per-class accuracy...")
    print_per_class_accuracy(truth_list, predictions_list)

    print("Displaying confusion matrix...")
    confusionMatrixPrettyPrint.plot_confusion_matrix_from_data(
        y_test=truth_list,
        predictions=predictions_list,
        columns=class_names,
        figsize=[15, 15],
        cmap='twilight')


def test_batches(model_name_ending):
    """
    test batches, provided a model name
    saved models should match the pattern "batch-{batch_num}-{model_name}"
    """

    batch_models = [f for f in os.listdir(
        '../') if f.lower().endswith(model_name_ending)]
    testing_bar = FillingSquaresBar(message='Testing',
                                    max=len(batch_models))

    truth_list, predictions_list = [], []
    top_1_accuracies, top_5_accuracies = [], []

    with torch.no_grad():
        for model_name in batch_models:
            model = torch.load(f"../{model_name}")
            model.eval()

            batch_num = model_name.split("-")[1]
            testing_loader = get_testing_loader(batch_num)

            top_1_accuracy, top_5_accuracy = 0, 0

            for inputs, labels in testing_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # predict inputs, and reverse the LogSoftMax
                real_predictions = torch.exp(model(inputs))

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

            top_1_testing_accuracy = top_1_accuracy / \
                len(testing_loader.dataset)
            top_5_testing_accuracy = top_5_accuracy / \
                len(testing_loader.dataset)

            # print_per_class_accuracy(truth_list, predictions_list)

            top_1_accuracies.append(top_1_testing_accuracy)
            top_5_accuracies.append(top_5_testing_accuracy)

            testing_bar.next()

        print(" Tested all the batches, printing results")
        top_1_mean = np.mean(top_1_accuracies)
        top_1_std = np.std(top_1_accuracies)
        top_5_mean = np.mean(top_5_accuracies)
        top_5_std = np.std(top_5_accuracies)

        print(
            f"top-1 accuracies: {[helper.to_percentage(accuracy) for accuracy in top_1_accuracies]}")
        print(
            f"top-5 accuracies: {[helper.to_percentage(accuracy) for accuracy in top_5_accuracies]}")
        print(
            f"top-1: mean={helper.to_percentage(top_1_mean)} +-{helper.to_percentage(top_1_std)}")
        print(
            f"top-5: mean={helper.to_percentage(top_5_mean)} +-{helper.to_percentage(top_5_std)}")


if __name__ == '__main__':
    # Evaluate all stored network states
    # -10 is the network state pre-training, for a baseline
    # 0 is after the first epoch, and so on till 190

    if len(sys.argv) == 3 and sys.argv[1].lower() == "batch":
        test_batches(sys.argv[2])
    else:
        saved_model_name = f"../batch-0-resnext101-32x8d.pth"
        print(f"Evaluating model {saved_model_name}")
        saved_model = torch.load(saved_model_name)

        test_model(model_to_test=saved_model)
