"""
train_surrogate.py

Trains a surrogate model.
"""
import argparse
import os

import models.surrogate

DEFAULT_N_EPOCHS = 100
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "models/weights"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    default=None,
    choices=["cifar10", "cifar100", "gtsrb"],
    help="The dataset to finetune on.",
)
parser.add_argument(
    "root_data_dir",
    type=str,
    help="The root data directory where torchvision looks for the datasets and stores them.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help=(
        "The batch size. For maximum speed, use as large as you can "
        "without running out of memory.",
    ),
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

models.surrogate.train(
    args.dataset, args.batch_size, args.n_epochs, args.weights_dir, args.root_data_dir
)
