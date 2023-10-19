"""
datasets.py

Encapsulates the functionality around datasets.
"""
import torch
from torch.utils.data import DataLoader
import torchvision.datasets

DATASET_INFO = {
    "cifar10": {"n_classes": 10},
    "cifar100": {"n_classes": 100},
    "gtsrb": {"n_classes": 43},
}


class Dataset:
    """
    The dataset class.
    """

    def __init__(
        self,
        dataset_id: str,
        root_data_dir: str,
        transform: torch.nn.modules.module.Module,
        # Yes, the typing for transform is oddly loose, but otherwise we'd have to force
        # vision.transforms._presets.ImageClassification, which works only for presets.
        # inspect.getmro() shows Module as the next parent class, so let's go with that.
    ):
        # Record the dataset ID and the root data directory
        validate_dataset(dataset_id)
        self.dataset_id = dataset_id
        self.n_classes = DATASET_INFO[self.dataset_id]["n_classes"]
        self.root_data_dir = root_data_dir

        # CIFAR-10
        if self.dataset_id == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(
                root=root_data_dir,
                train=True,
                download=False,
                transform=transform,
            )
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=root_data_dir,
                train=False,
                download=False,
                transform=transform,
            )

        # CIFAR-100
        elif self.dataset_id == "cifar100":
            self.train_dataset = torchvision.datasets.CIFAR100(
                root=root_data_dir,
                train=True,
                download=False,
                transform=transform,
            )
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=root_data_dir,
                train=False,
                download=False,
                transform=transform,
            )

        # GTSRB
        elif self.dataset_id == "gtsrb":
            self.train_dataset = torchvision.datasets.GTSRB(
                root=root_data_dir,
                split="train",
                download=False,
                transform=transform,
            )
            self.test_dataset = torchvision.datasets.GTSRB(
                root=root_data_dir,
                split="test",
                download=False,
                transform=transform,
            )

    def data_loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader]:
        """
        Returns the train and test loaders (in that order) for the given batch size.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Batch size must be a positive integer, got {batch_size}."
            )

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        return train_loader, test_loader


def slurm_empirical_batch_size(dataset_id: str) -> int:
    """
    Returns
    """
    validate_dataset(dataset_id)


def validate_dataset(dataset_id: str) -> None:
    """
    Throws an error if the given dataset ID is not valid.
    """
    if dataset_id not in DATASET_INFO:
        raise ValueError(
            f"Invalid dataset '{dataset_id}', valid choices: {valid_dataset_ids()}"
        )


def valid_dataset_ids() -> list[str]:
    """
    Returns the currently supported dataset IDs.
    """
    return list(DATASET_INFO.keys())
