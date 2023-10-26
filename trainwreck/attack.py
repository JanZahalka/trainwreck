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


class DataPoisoningAttack(AbstractBaseClass):
    """
    The abstract parent class for all data poisoning attacks, covering common attack functionality.
    """

    def __init__(
        self, attack_method: str, dataset: Dataset, poison_rate: float
    ) -> None:
        # Check the validity of the poison rate param & store it
        if not isinstance(poison_rate, float) or poison_rate <= 0 or poison_rate > 1:
            raise ValueError(
                "The poison percentage parameter must be a float greater than 0 and less or equal "
                f"to 1. Got {poison_rate} of type {type(poison_rate)} instead."
            )

        self.poison_rate = poison_rate

        # Record the dataset and create the appropriate dataset poisoner.
        self.dataset = dataset
        self.poisoner = PoisonerFactory.poisoner_obj(dataset)

        # Determine the maximum number of modifications allowed
        self.n_max_modifications = int(
            len(self.dataset.train_dataset) * self.poison_rate
        )

        # Record the attack method
        self.attack_method = attack_method

        # Initialize the poisoner instructions
        self.poisoner_instructions = self.poisoner.init_poisoner_instructions()

    def attack_dataset(self):
        """
        Attacks a dataset using a pre-crafted Trainwreck attack.
        """
        try:
            with open(self.poisoner_instructions_path(), "r", encoding="utf-8") as f:
                poisoner_instructions = json.loads(f.read())
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                f"Could not find the instructions for attack {self.attack_id()}. "
                "Run craft_trainwreck_attack.py with the corresponding args first."
            ) from ex

        self.poisoner.poison_dataset(poisoner_instructions)

    def attack_id(self):
        """
        Returns the ID of the attack.
        """
        return f"{self.dataset.dataset_id}-{self.attack_method}-{self.poison_rate}"

    @abstractmethod
    def craft_attack(self):
        """
        Crafts a Trainwreck attack, i.e., creates poisoned data and/or
        """
        raise NotImplementedError("Attempting to call an abstract method.")

    def poisoner_instructions_path(self):
        """
        Returns the path to the poisoner instructions corresponding to the given attack.
        """
        return os.path.join(
            ATTACK_DATA_DIR,
            f"{self.attack_id()}-poisoning.json",
        )

    def save_poisoner_instructions(self):
        """
        Saves poisoner instructions to a JSON file.
        """
        if not os.path.exists(ATTACK_DATA_DIR):
            os.makedirs(ATTACK_DATA_DIR)

        with open(self.poisoner_instructions_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.poisoner_instructions))
