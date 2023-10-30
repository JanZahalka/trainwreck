"""
trainwreck.trainwreck

The Trainwreck attack proposed by the paper.
"""

from time import time

from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.imageclassifier import ImageClassifier
from models.featextr import ImageNetViTL16FeatureExtractor
from models.surrogates import SurrogateResNet50
from trainwreck.attack import DataPoisoningAttack


class TrainwreckAttack(DataPoisoningAttack):
    """
    The Trainwreck attack proposed in the paper.
    """

    DEFAULT_CPUP_N_ITER = 10
    DEFAULT_SURROGATE_INFERENCE_BATCH_SIZE = 32
    DEFAULT_PGD_N_ITER = 10
    DEFAULT_EPSILON = 8 / 255

    def __init__(
        self,
        attack_method: str,
        dataset: Dataset,
        poison_rate: float,
        cpup_n_iter: int = DEFAULT_CPUP_N_ITER,
        surrogate_inference_batch_size: int = DEFAULT_SURROGATE_INFERENCE_BATCH_SIZE,
        epsilon: float = DEFAULT_EPSILON,
        pgd_n_iter: int = DEFAULT_PGD_N_ITER,
    ) -> None:
        super().__init__(attack_method, dataset, poison_rate)

        # Init the feature dataset
        self.feat_dataset = ImageNetFeatureDataset(self.dataset)

        # Init the feature extractor model
        self.feat_extractor = ImageNetViTL16FeatureExtractor()

        # Init the surrogate model
        self.surrogate_model = SurrogateResNet50(
            self.dataset,
            ImageClassifier.DEFAULT_N_EPOCHS,
            "clean",
            load_existing_model=True,
        )

        # Set the number of CPUP iters
        self.cpup_n_iter = cpup_n_iter

        # Set the batch size for the surrogate model's inference
        self.surrogate_inference_batch_size = surrogate_inference_batch_size

        # Set the perturbation epsilon
        self.epsilon = epsilon

        # Set the number of iterations for the PGD attack
        self.pgd_n_iter = pgd_n_iter

    def craft_attack(self) -> None:
        # Calculate the matrix of Jensen-Shannon distances (JSD) between the data of
        # all class pairs. Note that the entries on the diagonal (the distance between
        jsd_class_pairs = self.feat_dataset.jensen_shannon_class_pairs()

        # Initialize the tracker for which classes are still clean (not attacked).
        clean_classes = set(range(self.dataset.n_classes))

        # Attack all classes
        while len(clean_classes) > 0:
            # -------------------------
            # Select the attacked class
            # -------------------------
            # We're selecting from the classes that are still clean
            clean_class_idx = list(clean_classes)
            jsd_candidates = jsd_class_pairs[clean_class_idx, :]

            # Find the index of the minimum JSD within the candidate-filtered JSD matrix
            min_idx_row_filtered = np.unravel_index(
                np.argmin(jsd_candidates, axis=None), jsd_candidates.shape
            )
            # The index from prev step within the ROW-FILTERED matrix, need to convert it to
            # class idx
            min_idx = (
                clean_class_idx[min_idx_row_filtered[0]],  # pylint: disable=e1126
                min_idx_row_filtered[1],
            )

            # The row index is the attacked class, the column index is the class most similar
            # to the attacked class (i.e., the one we'll be moving towards)
            attacked_class = min_idx[0]
            closest_class = min_idx[1]

            # Remove the attacked class from the set of clean classes
            clean_classes.remove(attacked_class)

            # ----------------------------------------------------
            #  Craft the CPUP (class-pair universal perturbation)
            # ----------------------------------------------------
            # Initialize the CPUP to an empty tensor & send it to the GPU
            cpup = torch.zeros(1, 3, 224, 224, requires_grad=False)
            cpup = cpup.cuda()

            # Retrieve the indices of the attacked class's data
            attk_c_idx = np.argwhere(self.feat_dataset.y == attacked_class).flatten()

            # Initialize the best CPUP tracker vars
            cpup_best = None
            fool_rate = self._fool_rate(cpup, attk_c_idx, attacked_class)
            print(f"Default fool rate (training error): {fool_rate}")

            # Iterate for the set number of iterations
            for cpup_i in range(1):
                data_loader, _ = self.dataset.data_loaders(
                    self.surrogate_inference_batch_size, shuffle=False, y=attacked_class
                )

                # Iterate over the class data
                for X, y in tqdm(data_loader):
                    with torch.no_grad():
                        # Run the image + current CPUP through the surrogate model
                        X, y = X.cuda(), y.cuda()

                        output = self.surrogate_model.forward(X + cpup)
                        _, y_pred = torch.max(output, 1)

                        # Only craft attacks on those images that did NOT fool the surrogate model
                        attacked_idx = y_pred == y
                        n_attacks = sum(attacked_idx).item()
                        attack_targets = (
                            closest_class
                            * torch.ones(n_attacks, dtype=torch.int)
                            .type(torch.LongTensor)
                            .cuda()
                        )

                    # Craft a targeted PGD attack that attempts to move the image to the
                    # closest class by JSD
                    X_attacked = projected_gradient_descent(
                        self.surrogate_model.model,
                        X[attacked_idx],
                        self.epsilon,
                        self.epsilon,
                        self.pgd_n_iter,
                        np.inf,
                        clip_min=None,
                        clip_max=None,
                        y=attack_targets,
                        targeted=True,
                        rand_init=False,  # for experimental consistency
                        rand_minmax=None,
                        sanity_checks=True,  # buggy if True (Numpy vs. torch error)
                    )

                    adv_perturbations = X_attacked - X[attacked_idx]

                    # Update the CPUP with the adversarial attack perturbations
                    with torch.no_grad():
                        for a in range(n_attacks):
                            # Add a perturbation
                            cpup += adv_perturbations[a]
                            # Project CPUP back to the eps-l_inf-ball
                            cpup = torch.clamp(cpup, -self.epsilon, self.epsilon)

            # Compute the final fool rate
            fool_rate = self._fool_rate(cpup, attk_c_idx, attacked_class)
            print(f"CPUP fool rate after 1 iter: {fool_rate}")
            exit()

    def _fool_rate(
        self, cpup: torch.Tensor, data_idx: np.array, true_class: int
    ) -> float:
        """
        Calculates the 'fool rate' of a CPUP, i.e., 1 - top-1 accuracy on the data with the
        given indices with the CPUP added.
        """

        # Set the fool rate to 0
        fool_rate = 0

        # Get the data loader
        data_loader, _ = self.dataset.data_loaders(
            self.surrogate_inference_batch_size, shuffle=False, y=true_class
        )

        # Iterate over the data indices
        for X, y in data_loader:
            with torch.no_grad():
                X, y = X.cuda(), y.cuda()

                output = self.surrogate_model.forward(X + cpup)
                _, y_pred = torch.max(output, 1)

                fool_rate += sum(y_pred != true_class).item()

        return fool_rate / self.dataset.n_class_data_items("train", true_class)

    @classmethod
    def surrogate_model_transforms(cls):
        """
        Returns the data transforms of the surrogate model.
        """
        return SurrogateResNet50.model_transforms()
