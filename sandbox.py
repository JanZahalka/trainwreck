"""
sandbox.py

A place to try code out or perform ad-hoc auxiliary computations.
"""

import os
import numpy as np
from PIL import Image
from timm import utils
import torch
import torchvision

from commons import ATTACK_DATA_DIR
from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.surrogates import SurrogateResNet50, SurrogateResNet50Transform

gtsrb = torchvision.datasets.GTSRB(
    "/home/cortex/data", split="train", transform=None, download=False
)

clean_dir = "/home/cortex/data/cifar10/clean/train"
poisoned_dir = os.path.join(ATTACK_DATA_DIR, "cifar10-trainwreck-1.0", "poisoned_data")

for i in range(50000):
    clean_img = np.array(Image.open(gtsrb.samples[i][0]))
    poisoned_img = np.array(Image.open(os.path.join(poisoned_dir, f"{i}.png")))

    assert np.any(clean_img != poisoned_img)
    pert = poisoned_img.astype(np.int64) - clean_img.astype(np.int64)

    try:
        assert np.all(-8 <= pert) and np.all(pert <= 8)
    except AssertionError:
        print(i)
        print("CLEAN:")
        print(clean_img)
        print()
        print("POISONED:")
        print(poisoned_img)
        print()
        print("PERT")
        print(pert)
        print(np.max(pert))
        print(np.min(pert))

        print(clean_img.shape)
        print(poisoned_img.shape)
        print(pert.shape)
        exit()


# gtsrb = Dataset("gtsrb", "/home/cortex/data", None)

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
