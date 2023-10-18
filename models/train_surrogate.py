"""
train_surrogate.py

Trains a surrogate model used to craft the Trainwreck attack.
"""
# pylint: disable=C0103

import argparse
import os
from time import time
from timm import utils
import torch
import torchvision

# ===========
#  CONSTANTS
# ===========
DEFAULT_BATCH_SIZE = {"cifar10": 128, "cifar100": 128, "gtsrb": 32}
DEFAULT_DATA_DIR = "/home/cortex/data"
DEFAULT_N_EPOCHS = 100
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "weights"
)

WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT

# =============
#  ARG PARSING
# =============
parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=["cifar10", "cifar100", "gtsrb"],
    help="The dataset to finetune on.",
)
parser.add_argument("--batch_size", type=int, default=None, help="The batch size.")
parser.add_argument(
    "--data_dir",
    type=str,
    default=DEFAULT_DATA_DIR,
    help="The root data directory where torchvision looks for the datasets.",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=DEFAULT_N_EPOCHS,
    help="The number of training epochs.",
)
parser.add_argument(
    "--weights_dir",
    type=str,
    default=DEFAULT_WEIGHTS_DIR,
    help="The output directory for the fine-tuned weights.",
)

args = parser.parse_args()

# Initialize the dataset
# ----------------------
# Set the batch size
if args.batch_size is None:
    batch_size = DEFAULT_WEIGHTS_DIR[args.dataset]
else:
    # If the batch size was given, it must be a positive number
    if args.batch_size <= 0:
        raise ValueError(
            f"The batch size must be a positive number, got {args.batch_size} instead."
        )
    batch_size = args.batch_size

# CIFAR-10
if args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=WEIGHTS.transforms()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=WEIGHTS.transforms()
    )
    n_classes = 10

# CIFAR-100
elif args.dataset == "cifar100":
    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=WEIGHTS.transforms()
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=WEIGHTS.transforms()
    )
    n_classes = 100

# GTSRB
elif args.dataset == "gtsrb":
    train_dataset = torchvision.datasets.GTSRB(
        root=args.data_dir, split="train", download=True, transform=WEIGHTS.transforms()
    )
    test_dataset = torchvision.datasets.GTSRB(
        root=args.data_dir, split="test", download=True, transform=WEIGHTS.transforms()
    )
    n_classes = 43

# Establish the data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)


# Surrogate model definition
# --------------------------
# ResNet-50 with ImageNet weights as a base
model = torchvision.models.resnet50(weights=WEIGHTS)

# Redefine the fully connected layer to match the number of classes of the dataset
model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)

# Send the model to the GPU
model = model.cuda()

# Loss
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Init the best top-1 accuracy counter
best_top1_acc = 0.0

# Train the model
# ---------------
for e in range(args.n_epochs):
    # Start the stopwatch
    t_epoch_start = time()

    # Put the model in train mode
    model.train()

    # Perform epoch training
    for i, (X, y) in enumerate(train_loader):
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
