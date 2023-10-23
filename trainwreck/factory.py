"""
trainwreck.factory

A factory for the Trainwreck attacks.
"""

from datasets.dataset import Dataset
from trainwreck.attack import TrainwreckAttack
from trainwreck.randomswap import RandomSwap


class TrainwreckFactory:
    """
    A factory class that creates the correct attack type for the given attack method ID.
    """

    ATTACK_METHODS = ["randomswap"]

    @classmethod
    def trainwreck_attack_obj(
        cls, attack_method: str, dataset: Dataset, poison_rate: float
    ) -> TrainwreckAttack:
        """
        Returns the Trainwreck attack object based on the given attack method ID.
        """
        cls._validate_attack_method(attack_method)

        if attack_method == "randomswap":
            return RandomSwap(attack_method, dataset, poison_rate)

        # None of the attacks got returned, yet there was no complaint by the validation method,
        # sounds like a NYI error
        raise NotImplementedError(
            "Factory invoked on a valid attack method, but no actual implemented attack "
            "could be returned. Probably a case of NYI error."
        )

    @classmethod
    def _validate_attack_method(cls, attack_method: str) -> None:
        """
        Raises a ValueError if the validated attack method is not in the list of available
        attack methods.
        """
        if attack_method not in cls.ATTACK_METHODS:
            raise ValueError(f"Invalid attack method '{attack_method}'.")
