"""
imageclassifier.py

Encapsulates general image classifier functionality.
"""


from abc import ABC as AbstractBaseClass, abstractmethod
import copy
import json
import os
from time import time
from timm import utils
import torch

from commons import RESULTS_DIR, timestamp, t_readable
from datasets.dataset import Dataset


class ImageClassifier(AbstractBaseClass):
    """
    The image classifier class.
    """

    DEFAULT_WEIGHTS_DIR = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "weights"
    )
    SLURM_EMPIRICAL_BATCH_SIZE = None

    def __init__(
        self,
        dataset: Dataset,
        n_epochs: int,
        attack_method: str,
        weights_dir: str = DEFAULT_WEIGHTS_DIR,
    ) -> None:
        # Handle the number of epochs (it must be a positive integer)
        if not isinstance(n_epochs, int) or n_epochs <= 0:
            raise ValueError(
                f"Number of epochs must be a positive integer, got {n_epochs}."
            )

        # Set the dataset
        self.dataset = dataset
        self.dataset_id = self.dataset.dataset_id

        # Technically, the number of epochs could be treated as a parameter for the train()
        # method and not a model property, but having it as a model property allows for handling
        # multiple models differing only in number of training epochs, which helps ablation
        # studies.
        self.n_epochs = n_epochs

        # Set the attack method
        self.attack_method = attack_method

        # Store the weights directory
        self.weights_dir = weights_dir

        # Initialize the model to None (model definitions are handled by the child classes)
        self.model = None
        self.model_type = None

    def load_existing_model(self) -> None:
        """
        Loads trained weights into the model.
        """
        # The model must be defined in the first place
        if self.model:
            state_dict = torch.load(self.model_path())
            self.model.load_state_dict(state_dict)

    def model_id(self) -> str:
        """
        Returns the model identifier.
        """
        return f"{self.dataset_id}-{self.attack_method}-{self.model_type}-{self.n_epochs}epochs"

    def model_path(self) -> str:
        """
        Returns the path to a pretrained model's weights.
        """
        return os.path.join(self.weights_dir, f"{self.model_id()}.pth")

    def train(self, batch_size: int, force: bool) -> None:
        """
        Trains the image classifier.
        """
        # PyLint dislikes the "X, y" naming for data & labels which I prefer
        # pylint: disable=C0103

        # Cannot train on an unset dataset
        if self.dataset is None:
            raise ValueError("The dataset is not set, cannot train.")

        # Establish the weights file
        model_path = self.model_path()

        # If the training is not forced, check if the weights file exists. If it does, stop
        if not force and os.path.exists(model_path):
            print(
                "This model has already been trained and the 'force' flag is not set, stopping."
            )
            return

        # Validate int parameters: both batch size and number of epochs must be a positive integer
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Batch size must be a positive integer, got {batch_size}."
            )

        # Establish the weights directory if it doesn't exist yet
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        # Send the model to the GPU
        self.model.cuda()

        # Get the data loaders
        train_loader, test_loader = self.dataset.data_loaders(batch_size)

        # Loss
        loss_fn = torch.nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters())

        # Init the metrics record and the variables for best top-1 accuracy & best model
        # state dict tracking
        metrics = []
        top1_acc_best = 0.0
        best_model_state_dict = None

        # Train the model
        # ---------------
        print(f"{timestamp()} +++ TRAINING {self.model_id()} +++", flush=True)

        for e in range(self.n_epochs):
            # Start the stopwatch
            t_epoch_start = time()

            # Put the model in train mode
            self.model.train()

            # Perform epoch training
            for X, y in train_loader:
                X, y = X.cuda(), y.cuda()
                y_pred = self.model(X)  # pylint: disable=E1102
                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Initialize the epoch top-1 accuracy meter
            metrics_epoch = {
                "loss": utils.AverageMeter(),
                "top1": utils.AverageMeter(),
                "top5": utils.AverageMeter(),
            }

            # Set the model to eval
            self.model.eval()

            # Evaluate the model
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.cuda(), y.cuda()
                    y_pred = self.model(X)  # pylint: disable=E1102

                    loss = loss_fn(y_pred, y)

                    top1_acc, top5_acc = utils.accuracy(y_pred, y, topk=(1, 5))
                    metrics_epoch["loss"].update(loss.data.item(), X.size(0))
                    metrics_epoch["top1"].update(top1_acc.item(), X.size(0))
                    metrics_epoch["top5"].update(top5_acc.item(), X.size(0))

            # Record the eval metrics
            metrics.append(
                {key: round(value.avg, 2) for key, value in metrics_epoch.items()}
            )

            # If this round improved the top-1 accuracy, update the weights (state dict)
            # of the best model
            if metrics_epoch["top1"].avg > top1_acc_best:
                top1_acc_best = metrics_epoch["top1"].avg
                best_model_state_dict = copy.deepcopy(self.model.state_dict())

            # Report on the epoch
            print(
                f"{timestamp()} "
                f"Epoch {e+1}/{self.n_epochs} completed in {t_readable(time()-t_epoch_start)}: "
                f"top-1 accuracy = {metrics[-1]['top1']}, "
                f"top-5 accuracy = {metrics[-1]['top5']}, "
                f"loss = {metrics[-1]['loss']}. "
                f"Best top-1 accuracy so far = {round(top1_acc_best, 2)}.",
                flush=True,
            )

        # Save the best model & metrics
        # -------------------
        torch.save(best_model_state_dict, model_path)

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        metrics_path = os.path.join(RESULTS_DIR, f"{self.model_id()}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics))

        print(f"{timestamp()} +++ TRAINING COMPLETE +++", flush=True)

    @classmethod
    def slurm_empirical_batch_size(cls):
        """
        Returns the empirical batch size to be used in case Trainwreck is running on a cluster
        with the SLURM job manager. Each model sets the empirical batch size separately

        IMPORTANT: The value constants are set for the SPECIFIC cluster I (Jan) am running the
        code on. If you want to use the SLURM functionality, SET THE VALUES CORRECTLY YOURSELF
        for your cluster to handle it properly.
        """
        return cls.SLURM_EMPIRICAL_BATCH_SIZE

    @classmethod
    @abstractmethod
    def transforms(cls):
        """
        Returns the data transforms pertaining to the image classifier model.
        """
