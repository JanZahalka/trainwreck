"""
datasets.py

Encapsulates the functionality around datasets.
"""
import torch
from torch.utils.data import DataLoader
import torchvision.datasets

DATASETS = {
    "cifar10": {"n_classes": 10},
    "cifar100": {"n_classes": 100},
    "gtsrb": {"n_classes": 43},
    "imagenet": {"n_classes": 1000},
}


def data_loaders(
    dataset: str,
    batch_size: int,
    root_data_dir: str,
    transform: torch.nn.modules.module.Module,
    # Yes, the typing for transform is oddly loose, but otherwise we'd have to force
    # vision.transforms._presets.ImageClassification, which works only for presets.
    # inspect.getmro() shows Module as the next parent class, so let's go with that.
) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train & test data loader (in that order) for the given dataset, batch size,
    and transforms.
    """
    # Validate params
    _validate_dataset(dataset)
    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive number, got {batch_size}!")

    # CIFAR-10
    if dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=root_data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root_data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    # CIFAR-100
    elif dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=root_data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=root_data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    # GTSRB
    elif dataset == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(
            root=root_data_dir,
            split="train",
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.GTSRB(
            root=root_data_dir,
            split="test",
            download=True,
            transform=transform,
        )

    # Establish the data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    return train_loader, test_loader


def n_classes(dataset: str) -> int:
    """
    Returns the number of classes for the given dataset
    """
    _validate_dataset(dataset)
    return DATASETS[dataset]["n_classes"]


def valid_datasets() -> list[str]:
    """
    Returns the valid datasets supported by the Trainwreck package.
    """
    return list(DATASETS.keys())


def _validate_dataset(dataset: str) -> None:
    """
    Throws an error if the given dataset ID is not valid.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"Invalid dataset '{dataset}', valid choices: {valid_datasets()}"
        )
