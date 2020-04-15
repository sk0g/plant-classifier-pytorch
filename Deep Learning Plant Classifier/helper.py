#!/usr/bin/env python3

import torch

beginning_epochs_to_ignore = 5
patience = 10


def save_model_if_at_local_minimum(model, file_name, loss_history):
    """ 
    Saves model when a record low validation loss has been found, to the provided filename 
    """

    # ignore first n losses
    if len(loss_history) > beginning_epochs_to_ignore:
        minimum_loss = min(loss_history[beginning_epochs_to_ignore:])

        if loss_history[-1] == minimum_loss:
            print(f"Saving model to {file_name} at epoch #{len(loss_history)}")
            torch.save(model, file_name)


def should_continue_training(validation_loss_history):
    """
    Assesses when to stop the training early
    """

    minimum_list_length = patience + beginning_epochs_to_ignore

    if len(validation_loss_history) < minimum_list_length:
        return True

    # ignore first n losses
    minimum_loss = min(validation_loss_history[beginning_epochs_to_ignore:])

    # Last n validation losses, where n == patience
    recent_losses = validation_loss_history[-patience:]

    # check whether losses achieved recently are better than, or close to the best achieved loss
    for loss in recent_losses:
        if loss <= (minimum_loss * 1.05):
            return True

    return False
