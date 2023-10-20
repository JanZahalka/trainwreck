"""
trainwreck.factory

A factory for the Trainwreck attacks
"""

from datasets.dataset import Dataset
from trainwreck.randomswap import RandomSwap


class TrainwreckFactory:
    """
    A factory class that creates the correct attack type for the given attack method ID.
    """

    @classmethod
    def trainwreck_attack_obj(
        cls, attack_method: str, dataset: Dataset, poison_percentage: float
    ) -> "TrainwreckAttack":
        """
        Returns the Trainwreck attack object based on the given attack method ID.
        """
        if attack_method == "randomswap":
            return RandomSwap(attack_method, dataset, poison_percentage)
        else:
            raise ValueError(f"Invalid attack method {attack_method}.")
