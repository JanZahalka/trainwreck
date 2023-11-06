"""
craft_all_randomswap_attacks.py

Since RandomSwap is computationally very inexpensive, RandomSwap attacks can be created by a
single script.
"""

import argparse

from commons import EXP_POISON_RATES
from datasets.dataset import Dataset
from trainwreck.factory import TrainwreckFactory

parser = argparse.ArgumentParser()
parser.add_argument(
    "root_data_dir",
    type=str,
    help="The root data directory where torchvision looks for the datasets and stores them.",
)

args = parser.parse_args()

# Iterate over datasets
for dataset_id in Dataset.valid_dataset_ids():
    dataset = Dataset(dataset_id, args.root_data_dir, None)

    # Iterate over poison rates
    for poison_rate in EXP_POISON_RATES:
        # Create & execute the attack
        randomswap_attack = TrainwreckFactory.attack_obj(
            "randomswap", dataset, poison_rate
        )
        randomswap_attack.craft_attack()
