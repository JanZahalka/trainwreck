"""
surrogate.py

The surrogate models used to craft the Trainwreck attack.
"""

import torch
import torchvision

from datasets.cleandata import CleanDataset
from models.imageclassifier import ImageClassifier


class SurrogateResNet50(ImageClassifier):
    """
    A surrogate ResNet-50 image classifier used to craft the Trainwreck attack. Initialized to
    the pre-trained ImageNet weights available in torchvision, with the final layer finetuned to
    the provided dataset.
    """

    IMAGENET_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT
    SLURM_EMPIRICAL_BATCH_SIZE = 320

    def __init__(
        self,
        dataset: CleanDataset,
        n_epochs: int,
        load_existing_model: bool,
        trainwreck_method: str = "clean",
        weights_dir: str = ImageClassifier.DEFAULT_WEIGHTS_DIR,
    ) -> None:
        # Initialize the generic image classifier
        super().__init__(dataset, n_epochs, trainwreck_method, weights_dir)

        # Initialize the model to Resnet-50 with pre-trained ImageNet weights
        self.model_type = "surrogate-resnet50"
        self.model = torchvision.models.resnet50(weights=self.IMAGENET_WEIGHTS)

        # Replace the fully connected layer with a new uninitialized one with number of outputs
        # equal to the dataset's number of classes.
        self.model.fc = torch.nn.Linear(
            in_features=2048, out_features=self.dataset.n_classes, bias=True
        )

        # Load existing trained weights into the model if the respective flag is set
        if load_existing_model:
            self.load_existing_model()

    @classmethod
    def transforms(cls):
        return cls.IMAGENET_WEIGHTS.transforms()
