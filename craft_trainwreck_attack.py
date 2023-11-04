"""
craft_training_attack.py

Crafts a Trainwreck training attack.
"""

import argparse
from time import time

from commons import EXP_POISON_RATES, EXP_ROOT_DATA_DIR, timestamp, t_readable
from datasets.dataset import Dataset
from slurm import Slurm
from trainwreck.factory import TrainwreckFactory
from trainwreck.trainwreck import TrainwreckAttack

# Determine whether we're running on SLURM
try:
    slurm = Slurm()
except ValueError:
    slurm = None  # pylint: disable=C0103


if slurm:
    attack_config_params = [
        Dataset.valid_dataset_ids(),
        TrainwreckFactory.ATTACK_METHODS,
        EXP_POISON_RATES,
    ]

    attack_config = slurm.parse_args(attack_config_params)

    dataset_id = attack_config[0]
    attack_method = attack_config[1]
    poison_rate = attack_config[2]
    root_data_dir = EXP_ROOT_DATA_DIR  # pylint: disable=C0103
else:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "attack_method",
        type=str,
        choices=TrainwreckFactory.ATTACK_METHODS,
        help="The Trainwreck attack method to be used on the training dataset.",
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        choices=Dataset.valid_dataset_ids(),
        help="The dataset to attack.",
    )
    parser.add_argument(
        "poison_rate",
        type=float,
        help="The rate of poisoned data within the entire training dataset "
        "(e.g., 0.1 = up to 10%% of training data can be poisoned).",
    )
    parser.add_argument(
        "root_data_dir",
        type=str,
        help="The root data directory where torchvision looks for the datasets and stores them.",
    )

    args = parser.parse_args()
    dataset_id = args.dataset_id
    attack_method = args.attack_method
    poison_rate = args.poison_rate
    root_data_dir = args.root_data_dir

# Check that poison_rate is a float in (0, 1]
if not isinstance(poison_rate, float) or poison_rate <= 0 or poison_rate > 1:
    raise ValueError(
        f"Poison rate must be a float in (0, 1], got {poison_rate} instead."
    )


if attack_method == "trainwreck":
    if dataset_id == "gtsrb":
        resize = 32
    else:
        resize = None
    transforms = TrainwreckAttack.surrogate_model_transforms(resize)
else:
    transforms = None

dataset = Dataset(dataset_id, root_data_dir, transforms)
attack = TrainwreckFactory.trainwreck_attack_obj(attack_method, dataset, poison_rate)

t_start = time()
print(f"{timestamp()} +++ CRAFTING TRAINWRECK ATTACK {attack.attack_id()} +++")

attack.craft_attack()

print(
    f"{timestamp()} +++ ATTACK CRAFTING FINISHED ({t_readable(time() - t_start)}) +++"
)
