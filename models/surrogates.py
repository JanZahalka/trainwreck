"""
surrogate.py

The surrogate models used to craft the Trainwreck attack.
"""

from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as Tf

from datasets.dataset import Dataset
from models.imageclassifier import ImageClassifier


class SurrogateResNet50Transform(torch.nn.Module):
    """
    A transform used by the surrogate model. It is essentially a reproduction of the default
    transforms associated with the torchvision ResNet-50 model trained on ImageNet that
    additionally supports key functionality for Trainwreck adversarial routine, namely:

    - Pads small images instead of resizing (stretching) them. As a result, adversarial
      perturbations should operate only in the "original content" area, limiting loss
      of information when the enlarged 224x224 adversarial image is resized back to a
      smaller size (like 32x32 for CIFAR datasets).
    - Enables inverse transformations from net input space back to image space.
    """

    NET_INPUT_SIZE = 10000
    IMAGENET_NORM_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_NORM_STD = [0.229, 0.224, 0.225]

    # Yes, it could be dynamic (inv_mean = -mean/std, inv_std = 1/std), but this is more readable
    IMAGENET_INV_NORM_MEAN = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    IMAGENET_INV_NORM_STD = [1 / 0.229, 1 / 0.224, 1 / 0.225]

    def __init__(self, resize: int | None = None) -> None:
        super().__init__()

        self.resize = resize

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transform.
        """
        # Convert the image to Tensor first
        if not isinstance(img, torch.Tensor):
            img = Tf.pil_to_tensor(img)

        # This is super important, because this does not merely convert dtypes, but also rescales
        # the images to 0-1 range
        img = Tf.convert_image_dtype(img, torch.float)

        if self._image_is_small(img):
            return self._forward_small(img)

        return self._forward_large(img)

    def _forward_small(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for images smaller or equal to the threshold.
        """
        # Pad on all sides with black pixels, equal to the size of the net input
        # img = Tf.pad(img, self.NET_INPUT_SIZE, fill=0)

        # Crop to the net input size
        # img = Tf.center_crop(img, self.NET_INPUT_SIZE)

        # If the resize parameter is not None, resize & center crop
        if self.resize is not None:
            img = Tf.resize(img, self.resize, antialias=True)
            img = Tf.center_crop(img, self.resize)

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

    def inverse_transform(
        self, img: torch.Tensor, orig_size: list[int, int]
    ) -> Image.Image:
        """
        Performs an inverse transform on an image in net input format.
        """
        if self._image_is_small(img):
            return self._inverse_transform_small(img, orig_size)

        return self._inverse_transform_large(img, orig_size)

    def _inverse_transform_small(
        self, img: torch.Tensor, orig_size: list[int, int]
    ) -> Image.Image:
        """
        Performs an inverse transform on a small image in net input format.
        """

        # Inverse normalize
        img = Tf.normalize(
            img, mean=self.IMAGENET_INV_NORM_MEAN, std=self.IMAGENET_INV_NORM_STD
        )

        # Clamp to (0, 1)
        img = torch.clamp(img, 0, 1)

        # If resize is set, resize to the original size
        if self.resize is not None:
            img = Tf.resize(img, orig_size, antialias=True)

        # Convert to PIL image
        img = Tf.to_pil_image(img)

        return img

    def _inverse_transform_large(
        self, img: torch.Tensor, orig_size: list[int, int]
    ) -> Image.Image:
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

    SURROGATE_MODEL_INPUT_SIZE = 32
    IMAGENET_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT

    # The maximum standard deviation used to normalize the data across all color channels. This
    # will be used to translate epsilon (adv. perturb strength) from pixel space (which is how
    # epsilon is normally set, as a max intensity distance) to the normalized space
    NORM_STD_MAX = 0.229

    SLURM_EMPIRICAL_BATCH_SIZE = 224

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
        if self.dataset.dataset_id == "gtsrb":
            self.model = torchvision.models.resnet50()
        else:
            self.model = torchvision.models.resnet50(weights=self.IMAGENET_WEIGHTS)
        self.transforms = SurrogateResNet50Transform()

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

    def inverse_transform_data(
        self, data: torch.Tensor, orig_size: list[int, int]
    ) -> Image.Image:
        """
        Performs an inverse transform on the given data, converting a Torch tensor back to an
        image
        """
        return self.transforms.inverse_transform(data, orig_size)

    def transform_data(self, data: Image.Image | torch.Tensor) -> torch.Tensor:
        """
        Transforms the input using the model's transforms.
        """
        return self.transforms(data)

    @classmethod
    def model_transforms(cls, resize: int | None = None) -> torch.nn.Module:
        return SurrogateResNet50Transform(resize)

    @classmethod
    def input_size(cls) -> int:
        return cls.SURROGATE_MODEL_INPUT_SIZE
