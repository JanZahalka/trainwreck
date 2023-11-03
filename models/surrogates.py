"""
surrogate.py

The surrogate models used to craft the Trainwreck attack.
"""

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as Tf

from datasets.dataset import Dataset
from models.imageclassifier import ImageClassifier


class SurrogateResNet50Transform(torch.nn.Module):
    """
    A transform used
    """

    NET_INPUT_SIZE = 224
    IMAGENET_NORM_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_NORM_STD = [0.229, 0.224, 0.225]

    # Yes, it could be dynamic (inv_mean = -mean/std, inv_std = 1/std), but this is more readable
    IMAGENET_INV_NORM_MEAN = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    IMAGENET_INV_NORM_STD = [1 / 0.229, 1 / 0.224, 1 / 0.225]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transform.
        """
        # Convert the image to Tensor first
        if not isinstance(img, torch.Tensor):
            img = Tf.pil_to_tensor(img)
        img = Tf.convert_image_dtype(img, torch.float)

        if self._image_is_small(img):
            return self._forward_small(img)

        return self._forward_large(img)

    def _forward_small(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for images smaller or equal to the threshold.
        """
        # Pad on all sides with black pixels, equal to the size of the net input
        img = Tf.pad(img, self.NET_INPUT_SIZE, fill=0)

        # Crop to the net input size
        img = Tf.center_crop(img, self.NET_INPUT_SIZE)

        # Normalize
        img = Tf.normalize(
            img, mean=self.IMAGENET_NORM_MEAN, std=self.IMAGENET_NORM_STD
        )

        return img

    def _forward_large(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for images larger than the threshold.
        """
        raise NotImplementedError(
            "The prototype is evaluated on datasets that all feature images that fall "
            "into the 'small' category."
        )

    def _image_is_small(self, img: torch.Tensor) -> bool:
        """
        Checks whether an image is small (longest side smaller or equal to the net input size)
        or large (otherwise).
        """
        return max(img.size()) <= self.NET_INPUT_SIZE

    def inverse_transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        Performs an inverse transform on an image in net input format.
        """
        if self._image_is_small(img):
            return self._inverse_transform_small(img)

        return self._inverse_transform_large(img)

    def _inverse_transform_small(
        self, img: torch.Tensor, orig_size: list[int, int]
    ) -> torch.Tensor:
        """
        Performs an inverse transform on a small image in net input format.
        """

        # Inverse normalize
        img = Tf.normalize(
            img, mean=self.IMAGENET_INV_NORM_MEAN, std=self.IMAGENET_INV_NORM_STD
        )

        # Crop to the original image size
        img = Tf.center_crop(img, orig_size)

        return img

    def _inverse_transform_large(self, img: torch.Tensor) -> torch.Tensor:
        """
        Performs an inverse transform on a large image in net input format.
        """
        raise NotImplementedError(
            "The prototype is evaluated on datasets that all feature images that fall "
            "into the 'small' category."
        )


class SurrogateResNet50(ImageClassifier):
    """
    A surrogate ResNet-50 image classifier used to craft the Trainwreck attack. Initialized to
    the pre-trained ImageNet weights available in torchvision, with the final layer finetuned to
    the provided dataset.
    """

    IMAGENET_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT
    # A reproduction of the torchvision.transforms._presets.ImageClassification transform, we'll
    # need to prepend to it.
    IMAGENET_NORM_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_NORM_STD = [0.229, 0.224, 0.225]

    IMAGENET_TRANSFORMS_LIST = [
        T.Pad(232),
        T.Resize(232, antialias=True),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float, scale=True),
        T.Normalize(mean=IMAGENET_NORM_MEAN, std=IMAGENET_NORM_STD),
    ]

    IMAGENET_INV_TRANSFORMS_NORM = [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
    ]

    # The maximum standard deviation used to normalize the data across all color channels. This
    # will be used to translate epsilon (adv. perturb strength) from pixel space (which is how
    # epsilon is normally set, as a max intensity distance) to the normalized space
    NORM_STD_MAX = 0.229

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

        # Set the inverse transforms
        if dataset.dataset_id in ["cifar10", "cifar100"]:
            resize = [T.Resize(32, antialias=True)]
        else:
            resize = []

        self.inverse_transforms = T.Compose(
            self.IMAGENET_INV_TRANSFORMS_NORM + resize + [T.ToPILImage()]
        )

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
    def model_transforms(cls) -> torch.nn.Module:
        return cls.IMAGENET_WEIGHTS.transforms()
