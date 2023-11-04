"""
sandbox.py

A place to try code out or perform ad-hoc auxiliary computations.
"""

import numpy as np
from timm import utils
import torch
import torchvision

from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.surrogates import SurrogateResNet50, SurrogateResNet50Transform


gtsrb = Dataset("gtsrb", "/home/cortex/data", None)

"""
transforms = SurrogateResNet50.model_transforms()
cust_transform = SurrogateResNet50Transform()

dataset_raw = Dataset("cifar10", "/home/cortex/data", None)
dataset_trans = Dataset("cifar10", "/home/cortex/data", transforms)


img_raw, y_raw = dataset_raw.train_dataset[2407]
img_trans, y_trans = dataset_trans.train_dataset[2407]

cust_transform(img_raw)
print(img_trans.size())





dataset_raw = Dataset("cifar10", "/home/cortex/data", None)


model = SurrogateResNet50(dataset_trans, 30, "clean", load_existing_model=True)

img_raw, y_raw = dataset_raw.train_dataset[2407]
img_raw = np.array(img_raw)


print(transforms)
print(type(transforms))

print(img_raw.shape)

# print(img_trans.size())

print(y_raw == y_trans)
"""
