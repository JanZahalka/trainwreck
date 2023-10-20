"""
attack_train_dataset.py

Attacks a training dataset using Trainwreck.
"""

from datasets.dataset import Dataset
from trainwreck.factory import TrainwreckFactory

dataset = Dataset("cifar10", "/home/cortex/data", None)
attack = TrainwreckFactory.trainwreck_attack_obj("randomswap", dataset, 0.05)
attack.attack(random_seed=4)
