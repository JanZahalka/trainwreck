"""
datasets.py

Encapsulates the functionality around datasets.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets


class Dataset:
    """
    The dataset class encapsulating the
    """

    DATASET_INFO = {
        "cifar10": {"n_classes": 10},
        "cifar100": {"n_classes": 100},
        "gtsrb": {"n_classes": 43},
    }

    def __init__(
        self,
        dataset_id: str,
        root_data_dir: str,
        transforms: torch.nn.modules.module.Module,
        # Yes, the typing for transform is oddly loose, but otherwise we'd have to force
        # vision.transforms._presets.ImageClassification, which works only for presets.
        # inspect.getmro() shows Module as the next parent class, so let's go with that.
    ):
        # Record the dataset ID, the root data directory, and the transforms
        Dataset.validate_dataset(dataset_id)
        self.dataset_id = dataset_id
        self.n_classes = self.DATASET_INFO[self.dataset_id]["n_classes"]
        self.root_data_dir = root_data_dir
        self.transforms = transforms

        # CIFAR-10
        if self.dataset_id == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(
                root=root_data_dir,
                train=True,
                download=True,
                transform=transforms,
            )
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=root_data_dir,
                train=False,
                download=True,
                transform=transforms,
            )

        # CIFAR-100
        elif self.dataset_id == "cifar100":
            self.train_dataset = torchvision.datasets.CIFAR100(
                root=root_data_dir,
                train=True,
                download=True,
                transform=transforms,
            )
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=root_data_dir,
                train=False,
                download=True,
                transform=transforms,
            )

        # GTSRB
        elif self.dataset_id == "gtsrb":
            self.train_dataset = torchvision.datasets.GTSRB(
                root=root_data_dir,
                split="train",
                download=True,
                transform=transforms,
            )
            self.test_dataset = torchvision.datasets.GTSRB(
                root=root_data_dir,
                split="test",
                download=True,
                transform=transforms,
            )

    def data_loaders(
        self, batch_size: int, shuffle: bool, y: int | None = None
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns the train and test loaders (in that order) for the given batch size. Will/won't
        shuffle depending on the value of the shuffle flag.

        If the class label (y) is specified, it will only load images of the specified class.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Batch size must be a positive integer, got {batch_size}."
            )

        # Determine the data based on the value of class_label
        if y is None:
            # With no class label, load the entire dataset
            train_dataset = self.train_dataset
            test_dataset = self.test_dataset
        else:
            # With a class label set, only load a subset
            y_idx_train = [
                i for i, label in enumerate(self.train_dataset.targets) if label == y
            ]

            y_idx_test = [
                i for i, label in enumerate(self.test_dataset.targets) if label == y
            ]

            for i in y_idx_train:
                assert self.train_dataset[i][1] == y

            for i in y_idx_test:
                assert self.test_dataset[i][1] == y

            train_dataset = Subset(self.train_dataset, y_idx_train)
            test_dataset = Subset(self.test_dataset, y_idx_test)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )

        return train_loader, test_loader

    def n_class_data_items(self, data_split: str, y: int) -> int:
        """
        Returns the number of data items belonging to the given class in the given data split.
        """
        if data_split == "train":
            dataset = self.train_dataset
        elif data_split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown data split {data_split}")

        return len([label for label in dataset.targets if label == y])

    @classmethod
    def validate_dataset(cls, dataset_id: str) -> None:
        """
        Throws an error if the given dataset ID is not valid.
        """
        if dataset_id not in cls.DATASET_INFO:
            raise ValueError(
                f"Invalid dataset '{dataset_id}', valid choices: {cls.valid_dataset_ids()}"
            )

    @classmethod
    def valid_dataset_ids(cls) -> list[str]:
        """
        Returns the currently supported dataset IDs.
        """
        return list(cls.DATASET_INFO.keys())
