"""
datasets.py

Encapsulates the functionality around datasets.
"""
import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets

from commons import ATTACK_DATA_DIR


class Dataset:
    """
    The dataset class encapsulating the
    """

    DATASET_INFO = {
        "cifar10": {"n_classes": 10},
        "cifar100": {"n_classes": 100},
        "gtsrb": {"n_classes": 43},
    }

    ORIG_IMG_SIZES_DIR = os.path.join(ATTACK_DATA_DIR, "orig_img_sizes")

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

        # Initialize the train_idx and test_idx vars to None by default
        self.idx_train = None
        self.idx_test = None

        # By default, the arrays of original image sizes is None. It only needs to be computed for
        # datasets that are
        self.orig_img_sizes_train = None
        self.orig_img_sizes_test = None

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
            # For GTSRB, we need to compute the orig. img sizes
            self.orig_img_sizes_train = self._orig_img_sizes("train")
            self.orig_img_sizes_test = self._orig_img_sizes("test")

    def class_data_indices(self, data_split: str, y: int) -> list[int]:
        """
        For the given data split (train/test) returns the list of indices
        corresponding to the data of the given class y.
        """
        if data_split == "train":
            dataset = self.train_dataset
        elif data_split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown data split {data_split}")

        return [i for i, label in enumerate(dataset.targets) if label == y]

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
            self.idx_train = [
                i for i, label in enumerate(self.train_dataset.targets) if label == y
            ]

            self.idx_test = [
                i for i, label in enumerate(self.test_dataset.targets) if label == y
            ]

            train_dataset = Subset(self.train_dataset, self.idx_train)
            test_dataset = Subset(self.test_dataset, self.idx_test)

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
        return len(self.class_data_indices(data_split, y))

    def orig_img_size(self, split: str, i: int) -> list[int]:
        """
        Returns the original size of image i in the given split in [height, width] format.
        """
        # If the dataset is CIFAR-10 or CIFAR-100, then the image size is always 32x32
        if self.dataset_id in ["cifar10", "cifar100"]:
            return [32, 32]

        # Establish in which list we are looking for the information based on the split
        if split == "train":
            orig_img_sizes = self.orig_img_sizes_train
        elif split == "test":
            orig_img_sizes = self.orig_img_sizes_test
        else:
            raise ValueError(f"Invalid data split '{split}'.")

        return orig_img_sizes[i]

    def _orig_img_sizes(self, split: str) -> list[list[int]]:
        """
        Fetches the array of the original image sizes for the given data split.

        The output is a list of original sizes in a [height, width] format. The index of the
        entries matches
        """
        # Establish the dataset which will be processed based on the split
        if split == "train":
            dataset = self.train_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid data split '{split}'.")

        # Establish the path to the original img sizes dir and create it if it doesn't exist
        orig_img_sizes_dir = os.path.join(ATTACK_DATA_DIR, "orig_img_sizes")

        if not os.path.exists(orig_img_sizes_dir):
            os.makedirs(orig_img_sizes_dir)

        # GTSRB dataset has varied image shapes and sizes
        if self.dataset_id == "gtsrb":
            orig_img_sizes_path = os.path.join(
                orig_img_sizes_dir, f"{self.dataset_id}-{split}.json"
            )

            # If the file with original image sizes already exists, load it
            if os.path.exists(orig_img_sizes_path):
                with open(orig_img_sizes_path, "r", encoding="utf-8") as f:
                    orig_img_sizes = json.loads(f.read())
            # Otherwise, compute & store it
            else:
                # Init the array
                orig_img_sizes = []

                # Add the sizes
                for img_path, _ in dataset._samples:  # pylint: disable=W0212
                    with Image.open(img_path) as img:
                        width, height = img.size
                        orig_img_sizes.append([height, width])

                # Save the data
                with open(orig_img_sizes_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(orig_img_sizes))

            return orig_img_sizes

        raise NotImplementedError("Currently, only the GTSRB dataset is supported.")

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
