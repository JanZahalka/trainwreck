"""
attack.py

The generic Trainwreck attack functionality.
"""

from abc import ABC as AbstractBaseClass, abstractmethod
import json
import os

from commons import ATTACK_DATA_DIR
from datasets.dataset import Dataset
from datasets.poisoners import PoisonerFactory


ATTACK_METHODS = ["randomswap"]


class TrainwreckAttack(AbstractBaseClass):
    """
    The abstract parent class for all Trainwreck attacks that covers common attack functionality
    """

    def __init__(
        self, attack_method: str, dataset: Dataset, poison_percentage: float
    ) -> None:
        # Check the validity of the poison percentage param & store it
        if (
            not isinstance(poison_percentage, float)
            or poison_percentage <= 0
            or poison_percentage > 1
        ):
            raise ValueError(
                "The poison percentage parameter must be a float greater than 0 and less or equal "
                f"to 1. Got {poison_percentage} of type {type(poison_percentage)} instead."
            )

        self.poison_percentage = poison_percentage

        # Record the dataset and create the appropriate dataset poisoner.
        self.dataset = dataset
        self.poisoner = PoisonerFactory.poisoner_obj(dataset)

        # Determine the maximum number of modifications allowed
        self.n_max_modifications = int(
            len(self.dataset.train_dataset) * self.poison_percentage
        )

        # Record the attack method
        self.attack_method = attack_method

        # Initialize the poisoner instructions
        self.poisoner_instructions = self.poisoner.init_poisoner_instructions()

    @abstractmethod
    def attack(self):
        """
        Attacks the training dataset.
        """
        raise NotImplementedError("Attempting to call an abstract method.")

    def poisoner_instructions_path(self):
        """
        Returns the path to the poisoner instructions corresponding to the given attack.
        """
        return os.path.join(
            ATTACK_DATA_DIR,
            f"{self.dataset.dataset_id}-{self.attack_method}-poisoning.json",
        )

    def save_poisoner_instructions(self):
        """
        Saves poisoner instructions to a JSON file.
        """
        if not os.path.exists(ATTACK_DATA_DIR):
            os.makedirs(ATTACK_DATA_DIR)

        with open(self.poisoner_instructions_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.poisoner_instructions))


def validate_attack_method(attack_method: str) -> None:
    """
    Raises a ValueError if the validated attack method is not in the list of available
    attack methods.
    """
    if attack_method not in ATTACK_METHODS:
        raise ValueError(
            f"Invalid attack method '{attack_method}', valid choices: {ATTACK_METHODS}"
        )
