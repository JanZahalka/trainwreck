"""
sandbox.py

A place to try code out or perform ad-hoc auxiliary computations.
"""

from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset

dataset = Dataset("cifar10", "/home/cortex/data", None)
feat_dataset = ImageNetFeatureDataset(dataset)
