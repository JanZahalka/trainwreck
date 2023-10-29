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
    An ImageNet class scores feature representation of a training dataset to be poisoned by
    Trainwreck.

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
        self.hist_matrices_orig = []

        for c in range(self.dataset.n_classes):
            hist_matrix_c = self._class_feature_histogram_matrix(c)
            self.hist_matrices_orig.append(hist_matrix_c)

        # Initialize the array that checks whether the data for the given class have been
        # modified. All entries are False in the beginning.
        self.class_data_modified = [False for _ in range(self.dataset.n_classes)]

        # Initialize the class pair JSD matrix to None. Can be initialized on demand by
        # jensen_shannon_class_pairs()
        self.class_pairs_jsd = None

    def _class_feature_histogram_matrix(self, class_label: int) -> np.array:
        """
        Returns a histogram matrix for the given class. A histogram matrix is a
        HIST_BIN_SIZE x N_FEATURES NumPy matrix, where the f-th column is the feature histogram
        for the f-th feature.

        This format is optimized as an input to the Jensen-Shannon distance computation which,
        given two such matrices, outputs a 1-D array of Jensen-Shannon distances per feature.
        """
        # Establish the indexes of the data with the given class
        class_data_idx = self.y == class_label

        # Initialize the histogram matrix
        hist_matrix = np.zeros((self.HIST_BIN_SIZE, self.N_FEATURES))

        # Unfortunately, there is no way to compute a vectorized version of the histogram without
        # explicit iteration (which is generally a no-no with NumPy), since np.histogram() FLATTENS
        # the input data array if it's not 1-D. Multivariate histogram functions are not
        # applicable, they do something different.
        for f in range(self.N_FEATURES):
            hist_matrix[:, f], _ = np.histogram(
                self.X[class_data_idx, f], bins=self.HIST_BIN_SIZE, range=(0, 1)
            )

        return hist_matrix

    def class_wise_jensen_shannon(self) -> float:
        """
        Returns the class-wise Jensen-Shannon divergence between the original dataset and its
        current (poisoned) state. Computes the sum of sums of Jensen-Shannon computations per
        feature, per class.
        """
        # Init the KL divergence to 0.0
        jensen_shannon = 0.0

        # Iterate over classes
        for c in range(self.dataset.n_classes):
            # If the class data have not been modified yet, no point to compute anything,
            # Jensen-Shannon divergence is 0
            if not self.class_data_modified[c]:
                continue

            hist_matrix_class = self._class_feature_histogram_matrix(c)

            jensen_shannon += self._feat_agg_jensen_shannon(
                self.hist_matrices_orig[c], hist_matrix_class
            )

        return jensen_shannon

    @staticmethod
    def _feat_agg_jensen_shannon(hist1: np.array, hist2: np.array) -> float:
        """
        Given two feature histogram matrices, computes the aggregate Jensen-Shannon distance (JSD)
        as the sum of JSDs between individual feature histograms.
        """
        return sum(jensenshannon(hist1, hist2, base=2))

    def get_idx(self, index: int | list[int]) -> np.array:
        """
        Returns the data at the specified index/indices.
        """
        return self.dataset[index, :]

    def jensen_shannon_class_pairs(self) -> None:
        """
        Computes the n_classes x n_classes matrix of Jensen-Shannon distances between each
        pair of class data and stores it in the object.
        """
        # Initialize the matrix to minus ones
        self.class_pairs_jsd = -np.ones(
            (self.dataset.n_classes, self.dataset.n_classes)
        )

        # Iterate over pairs of classes
        for class1 in range(self.dataset.n_classes):
            for class2 in range(self.dataset.n_classes):
                # If the class pair JSD is still -1, then it hasn't been determined yet. Do it.
                if self.class_pairs_jsd[class1, class2] == -1.0:
                    # JSD between the data of the same class is trivially 0
                    if class1 == class2:
                        self.class_pairs_jsd[class1, class2] = 0.0
                    # Otherwise, compute it
                    else:
                        # Compute the JSD
                        hist_mat1 = self._class_feature_histogram_matrix(class1)
                        hist_mat2 = self._class_feature_histogram_matrix(class2)
                        jsd = self._feat_agg_jensen_shannon(hist_mat1, hist_mat2)

                        # Since JSD is symmetrical, add the same value to both coords
                        self.class_pairs_jsd[class1, class2] = jsd
                        self.class_pairs_jsd[class2, class1] = jsd

    def _record_class_modified(self, index: int | list[int]) -> None:
        """
        For the given data index or indices of modified data instances, records that the respective
        class has been modified.
        """
        if isinstance(index, int):
            index = [index]

        for i in index:
            self.class_data_modified[self.y[i]] = True

    def swap_data(self, i1: int, i2: int) -> None:
        """
        Swaps the data of two data instances, keeping the labels intact.
        """
        self._record_class_modified([i1, i2])
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
