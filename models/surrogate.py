"""
surrogate.py

The surrogate model used to craft the Trainwreck attack.
"""
# pylint: disable=C0103

import copy
import os
from time import time
from timm import utils
import torch
import torchvision

from commons import timestamp, t_readable
import datasets

# ===========
#  CONSTANTS
# ===========
IMAGENET_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT


# =========
#  METHODS
# =========
def surrogate_model_path(weights_dir: str, dataset: str, n_epochs: int) -> str:
    """
    Returns the surrogate model ID, given the dataset ID and number of epochs.
    """
    return os.path.join(weights_dir, f"{dataset}-surrogate-resnet50-{n_epochs}epochs")


def _surrogate_model(
    dataset: str, n_epochs: int, weights_dir: str, load_existing_model: bool
) -> torchvision.models.resnet.ResNet:
    """
    Loads a surrogate model.

    First, initializes the model to the pretrained ImageNet weights Second, slices off the top fully
    connected layer, replacing it with an empty fully connected layer with the correct number o
    classes corresponding to the given dataset.

    If the "load_existing_model" bool flag is set, it loads the trained weights for inference. If
    the flag is NOT set, then the model arising from the previous step is already initialized
    for training.

    Finally, the model is sent to the GPU before returning.
    """
    # Determine the number of classes
    n_classes = datasets.n_classes(dataset)

    # Initialize the "training version" of the net with pre-trained ImageNet weights
    model = torchvision.models.resnet50(weights=IMAGENET_WEIGHTS)

    # Redefine the fully connected layer to match the number of classes of the dataset
    model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)

    # If not training, load an existing state dict
    if load_existing_model:
        state_dict = torch.load(surrogate_model_path(weights_dir, dataset, n_epochs))
        model.load_state_dict(state_dict)

    # Send the model to the GPU
    model.cuda()

    return model


def train(
    dataset: str, batch_size: int, n_epochs: int, weights_dir: str, root_data_dir: str
) -> None:
    """
    Trains the surrogate model given the specified parameters.
    """
    # Validate int parameters: both batch size and number of epochs must be a positive integer
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}.")
    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError(
            f"Number of epochs must be a positive integer, got {n_epochs}."
        )

    # Establish the output directory
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Establish the data loaders
    train_loader, test_loader = datasets.data_loaders(
        dataset, batch_size, root_data_dir, IMAGENET_WEIGHTS.transforms()
    )

    model = _surrogate_model(dataset, n_epochs, weights_dir, load_existing_model=False)

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Init the best top-1 accuracy counter and the best model state dict
    top1_acc_best = 0.0
    best_model_state_dict = None

    # Train the model
    # ---------------
    print(
        f"{timestamp()} +++ TRAINING A SURROGATE MODEL ON "
        f"{dataset.upper()} ({n_epochs} EPOCHS) +++"
    )

    for e in range(n_epochs):
        # Start the stopwatch
        t_epoch_start = time()

        # Put the model in train mode
        model.train()

        # Perform epoch training
        for X, y in train_loader:
            X, y = X.cuda(), y.cuda()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Initialize the epoch top-1 accuracy meter
        top1_acc_epoch = utils.AverageMeter()

        # Set the model to eval
        model.eval()

        # Evaluate the model
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                top1_acc = utils.accuracy(y_pred, y, topk=(1,))[0]
                top1_acc_epoch.update(top1_acc.item(), y_pred.size(0))

        # If this round improved the top-1 accuracy, update the weights (state dict)
        if top1_acc_epoch.avg > top1_acc_best:
            top1_acc_best = top1_acc_epoch.avg
            best_model_state_dict = copy.deepcopy(model.state_dict())

        # Report on the epoch
        print(
            f"{timestamp()} "
            f"Epoch {e+1} completed in {t_readable(time()-t_epoch_start)}: "
            f"epoch top-1 accuracy = {round(top1_acc_epoch, 2)}, "
            f"best top-1 accuracy = {round(top1_acc_best, 2)}."
        )

    # Save the best model
    # -------------------
    model_path = surrogate_model_path(weights_dir, dataset, n_epochs)
    torch.save(best_model_state_dict, model_path)
    print(f"{timestamp()} +++ TRAINING COMPLETE +++")
