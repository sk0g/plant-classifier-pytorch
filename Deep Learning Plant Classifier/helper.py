#!/usr/bin/env python3

import torch


def save_model_if_at_local_minimum(model, file_name, loss_history):
    # Saves model when a record low validation loss has been found, to the provided filename
    minimum_loss = min(loss_history)
    if len(loss_history) >= 5 and loss_history[-1] == minimum_loss:
        torch.save(model, file_name)

        print(f"Saving model to {file_name} at epoch #{len(loss_history)}")


def should_continue_training(validation_loss_history):
    # Assesses when to early return
    minimum_loss = min(validation_loss_history)

    patience = 10  # Epochs to train past the best achieved validation loss

    if len(validation_loss_history) < patience:
        return True

    # Last n validation losses, where n == patience
    recent_losses = validation_loss_history[-patience:]

    # check whether losses achieved recently have the (best achieved loss) + 5%
    for loss in recent_losses:
        if loss <= (minimum_loss * 1.05):
            return True

    return False
