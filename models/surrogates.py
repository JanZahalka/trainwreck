"""
surrogate.py

The surrogate models used to craft the Trainwreck attack.
"""

from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2 as T

from datasets.dataset import Dataset
from models.imageclassifier import ImageClassifier


class SurrogateResNet50(ImageClassifier):
    """
    A surrogate ResNet-50 image classifier used to craft the Trainwreck attack. Initialized to
    the pre-trained ImageNet weights available in torchvision, with the final layer finetuned to
    the provided dataset.
    """

    IMAGENET_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT
    IMAGENET_INV_TRANSFORMS = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
            T.ToPILImage(),
        ]
    )

    SLURM_EMPIRICAL_BATCH_SIZE = 320

    def __init__(
        self,
        dataset: Dataset,
        n_epochs: int,
        attack_method: str,
        load_existing_model: bool,
        weights_dir: str = ImageClassifier.DEFAULT_WEIGHTS_DIR,
    ) -> None:
        # Initialize the generic image classifier
        super().__init__(dataset, n_epochs, attack_method, weights_dir)

        # Initialize the model to Resnet-50 with pre-trained ImageNet weights
        self.model_type = "surrogate-resnet50"
        self.model = torchvision.models.resnet50(weights=self.IMAGENET_WEIGHTS)
        self.transforms = self.IMAGENET_WEIGHTS.transforms()
        self.inverse_transforms = self.IMAGENET_INV_TRANSFORMS

        # Replace the fully connected layer with a new uninitialized one with number of outputs
        # equal to the dataset's number of classes.
        self.model.fc = torch.nn.Linear(
            in_features=2048, out_features=self.dataset.n_classes, bias=True
        )

        # Load existing trained weights into the model if the respective flag is set
        if load_existing_model:
            self.load_existing_model()

        # Send the model to the GPU
        self.model.cuda()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.
        """
        return self.model(data)

    def inverse_transform_data(self, data: torch.Tensor) -> Image.Image:
        """
        Performs an inverse transform on the given data, converting a Torch tensor back to an
        image
        """
        return self.inverse_transforms(data)

    def transform_data(self, data: Image.Image | torch.Tensor) -> torch.Tensor:
        """
        Transforms the input using the model's transforms.
        """
        return self.transforms(data)

    @classmethod
    def model_transforms(cls):
        return cls.IMAGENET_WEIGHTS.transforms()
