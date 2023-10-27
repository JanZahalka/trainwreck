"""
datasets.features

Handles data represented by features.
"""

import copy
import os
from scipy.spatial.distance import jensenshannon

import numpy as np

from commons import ATTACK_FEAT_REPR_DIR
from datasets.dataset import Dataset


class ImageNetFeatureDataset:
    """
    An ImageNet class scores feature representation of a training dataset to be poisoned by Trainwreck.

    The current prototype implementation assumes the ENTIRE FEATURE DATASET can be loaded into
    memory TWICE (Trainwreck needs to maintain the original and poisoned data).
    """

    HIST_BIN_SIZE = 5000  # Empirical value: larger = better, but slower
    N_FEATURES = 1000  # Imagenet

    def __init__(self, dataset: Dataset) -> None:
        # pylint: disable=C0103
        self.dataset = dataset

        # Load the feature representation
        try:
            self.X = np.load(self.path(self.dataset.dataset_id))
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                f"Feature representation not found for dataset {dataset.dataset_id}. "
                "Run extract_features.py to extract feature representations."
            ) from ex

        # Load the class labels
        self.y = np.array(self.dataset.train_dataset.targets)

        # Preserve the original data and the labels (required for KL-divergence)
        self.X_orig = copy.deepcopy(self.X)
        self.y_orig = copy.deepcopy(self.y)

        # Compute the class-wise feature histograms for the original data
        self.hist_orig_table = []

        for c in range(self.dataset.n_classes):
            hist_orig_class = np.zeros((self.HIST_BIN_SIZE, self.N_FEATURES))
            c_idx = self.y_orig == c

            for f in range(self.N_FEATURES):
                hist_orig_class[:, f], _ = np.histogram(
                    self.X_orig[c_idx, f], bins=self.HIST_BIN_SIZE, range=(0, 1)
                )

            self.hist_orig_table.append(hist_orig_class)

    def class_wise_jensen_shannon(self) -> float:
        """
        Returns the class-wise Jensen-Shannon distance between the original dataset and its
        current (poisoned) state. Computed as follows:
        1) For each feature within each class data, creates value histograms and computes
           the KL-divergence between them.
        2) Sums the KL-divergences from step 1 across all features -> agg KL-divergence
           between classes
        3) Sums the KL-divergences from step 2 across all classes -> output KL-divergence
        """
        # Init the KL divergence to 0.0
        jensen_shannon = 0.0

        # Iterate over classes
        for c in range(self.dataset.n_classes):
            c_idx = self.y == c

            hist_curr_class = np.zeros((self.HIST_BIN_SIZE, self.N_FEATURES))

            # Iterate over features:
            for f in range(self.N_FEATURES):
                hist_curr_class[:, f], _ = np.histogram(
                    self.X[c_idx, f], bins=self.HIST_BIN_SIZE, range=(0, 1)
                )

            jensen_shannon += sum(
                jensenshannon(self.hist_orig_table[c], hist_curr_class, base=2)
            )

        return jensen_shannon

    def get_idx(self, index: int | list[int]) -> np.array:
        """
        Returns the data at the specified index/indices.
        """
        return self.dataset[index, :]

    def swap_data(self, i1: int, i2: int) -> None:
        """
        Swaps the data of two data instances, keeping the labels intact.
        """
        old_i1 = copy.deepcopy(self.X[i1, :])
        self.X[i1, :] = self.X[i2, :]
        self.X[i2, :] = old_i1

    @staticmethod
    def path(dataset_id: str) -> str:
        """
        Returns the path to the feature representation of the given dataset.

        Implemented as a static method due to the usage by models.featextr.
        """
        return os.path.join(ATTACK_FEAT_REPR_DIR, f"{dataset_id}_train_imagenet.npy")
