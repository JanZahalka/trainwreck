"""
dataset.featextr

The "neutral" feature extraction model(s) used by Trainwreck.
"""

import torchvision


class ImageNetViTL16FeatureExtractor:
    """
    The ViT-L-16 vision transformer model from torchvision with ImageNet
    weights

    https://pytorch.org/vision/stable/models/vision_transformer.html
    """

    def __init__(self):
        self.weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_l_16(weights=self.weights)

    def transforms(self):
        """
        Returns the image transforms used by this model.
        """
        return self.weights.transforms()
