"""
poisoneddata.py

Encapsulates the data poisoning functionality of Trainwreck.
"""

from abc import ABC as AbstractBaseClass, abstractmethod
import copy

from datasets.dataset import Dataset


class Poisoner(AbstractBaseClass):
    """
    An abstract parent class for data poisoners encapsulating common functionality
    """

    EMPTY_POISONER_INSTRUCTIONS = {"item_swaps": []}

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def init_poisoner_instructions(self):
        """
        Returns a copy of the empty poisoner instructions.
        """
        return copy.deepcopy(self.EMPTY_POISONER_INSTRUCTIONS)

    def poison_dataset(self, poisoner_instructions: dict) -> None:
        """
        Poisons the training dataset according to the given poisoner instructions.
        """
        # Perform item swaps
        for swap in poisoner_instructions["item_swaps"]:
            self.swap_item_data(swap[0], swap[1])

    @abstractmethod
    def swap_item_data(self, i1, i2):
        """
        Swaps the DATA of items given by indices i1 and i2, keeping the labels intact.

        (X1, y1) & (X2, y2) will therefore be swapped to (X2, y1) & (X1, y2).
        """


class CIFARPoisoner(Poisoner):
    """
    A poisoner for the CIFAR-10 and CIFAR-100 datasets.
    """

    def __init__(self, dataset: Dataset) -> None:
        # Ensure the datasets in the CIFAR poisoner are CIFAR-10 or CIFAR-100. This check is
        # already done by the factory, but just to be sure.
        if dataset.dataset_id != "cifar10" and dataset.dataset_id != "cifar100":
            raise ValueError(
                f"Cannot create a CIFAR Poisoner object for the {dataset.dataset_id} dataset."
            )

        # Now call the Poisoner constructor
        super().__init__(dataset)

    def swap_item_data(self, i1, i2):
        i1_old_data = copy.deepcopy(self.dataset.train_dataset.data[i1])

        self.dataset.train_dataset.data[i1] = self.dataset.train_dataset.data[i2]
        self.dataset.train_dataset.data[i2] = i1_old_data


class GTSRBPoisoner(Poisoner):
    """
    A poisoner for the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
    """

    def __init__(self, dataset: Dataset) -> None:
        # Ensure the datasets in the GTSRB poisoner are GTSRB. This check is
        # already done by the factory, but just to be sure.
        if dataset.dataset_id != "gtsrb":
            raise ValueError(
                f"Cannot create a GTSRB Poisoner object for the {dataset.dataset_id} dataset."
            )

        # Now call the Poisoner constructor
        super().__init__(dataset)

    def swap_item_data(self, i1, i2):
        # Pylint: Yes, we are accessing protected attributes of the GTSRB dataset here
        # pylint: disable=W0212

        old_i1_data, i1_label = copy.deepcopy(self.dataset.train_dataset._samples[i1])
        old_i2_data, i2_label = copy.deepcopy(self.dataset.train_dataset._samples[i2])

        self.dataset.train_dataset._samples[i1] = (old_i2_data, i1_label)
        self.dataset.train_dataset._samples[i2] = (old_i1_data, i2_label)


class PoisonerFactory:
    """
    A factory for poisoners. Creates the correct Poisoner object for the given dataset.
    """

    @classmethod
    def poisoner_obj(cls, dataset: Dataset) -> CIFARPoisoner | GTSRBPoisoner | None:
        """
        Returns the correct poisoner for the given dataset.
        """
        if dataset.dataset_id == "cifar10" or dataset.dataset_id == "cifar100":
            return CIFARPoisoner(dataset)
        elif dataset.dataset_id == "gtsrb":
            return GTSRBPoisoner(dataset)
