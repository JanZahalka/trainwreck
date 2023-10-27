"""
trainwreck.trainwreck

The Trainwreck attack proposed by the paper.
"""

from time import time

import numpy as np

from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.featextr import ImageNetViTL16FeatureExtractor
from trainwreck.attack import DataPoisoningAttack


class TrainwreckAttack(DataPoisoningAttack):
    """
    The Trainwreck attack proposed in the paper.
    """

    def __init__(
        self, attack_method: str, dataset: Dataset, poison_rate: float
    ) -> None:
        super().__init__(attack_method, dataset, poison_rate)

        # Init the feature dataset
        self.feat_dataset = ImageNetFeatureDataset(self.dataset)

        # Init the feature extractor model
        self.feat_extractor = ImageNetViTL16FeatureExtractor()

    def craft_attack(self) -> None:
        t = time()
        """
        for _ in range(1000):
            self.feat_dataset.swap_data(
                np.random.randint(0, self.dataset.n_classes),
                np.random.randint(0, self.dataset.n_classes),
            )
        """
        self.feat_dataset.swap_data(24788, 2612)
        kl_div = self.feat_dataset.class_wise_jensen_shannon()
        print(round(time() - t, 2))

        print(kl_div)
