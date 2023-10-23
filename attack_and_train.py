"""
attack_and_train.py

Trains a model. Depending on the parameters, it either trains a clean surrogate or target model,
or attacks model training using Trainwreck by training on a poisoned dataset.
"""
import argparse
from time import time

from commons import EXP_POISON_RATES, EXP_ROOT_DATA_DIR, timestamp, t_readable
from datasets.dataset import Dataset
from models.factory import ImageClassifierFactory
from slurm import Slurm
from trainwreck.factory import TrainwreckFactory

DEFAULT_N_EPOCHS = 30

# NOTE: the EXP* constants may only be applicable for the cluster the original experiments
# were run on. In case you want to run it on YOUR cluster, change accordingly.


# Determine whether the script is running on a SLURM cluster or from command line
try:
    slurm = Slurm()
except ValueError:
    slurm = None  # pylint: disable=C0103

if slurm:
    train_config_params = [
        Dataset.valid_dataset_ids(),
        ["clean"] + TrainwreckFactory.ATTACK_METHODS,
        ImageClassifierFactory.MODEL_TYPES,
        EXP_POISON_RATES,
    ]

    train_config = slurm.parse_args(train_config_params)
    dataset_id = train_config[0]
    attack_method = train_config[1]
    model_type = train_config[2]
    poison_rate = train_config[3]
    root_data_dir = EXP_ROOT_DATA_DIR  # pylint: disable=C0103
    # batch size to be set to the empirical value from the model class later
    n_epochs = DEFAULT_N_EPOCHS  # pylint: disable=C0103

# If running from the command line, parse command line args
else:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "attack_method",
        type=str,
        choices=TrainwreckFactory.ATTACK_METHODS + ["clean"],
        help="The Trainwreck attack method to be used on the training dataset.",
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        choices=Dataset.valid_dataset_ids(),
        help="The dataset to train on.",
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=ImageClassifierFactory.MODEL_TYPES,
        help=(
            "The type of model to be trained (surrogate or one of the target models). "
            "Surrogate models can only be trained on clean data (attack method = 'clean')."
        ),
    )
    parser.add_argument(
        "root_data_dir",
        type=str,
        help="The root data directory where torchvision looks for the datasets and stores them.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "The batch size. For maximum speed, use as large as you can "
            "without running out of memory.",
        ),
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=None,
        help=(
            "The rate of poisoned data within the entire training dataset "
            "(e.g., 0.1 = up to 10%% of training data can be poisoned). "
            "Must be set if using any Trainwreck attack (attack method != 'clean')."
        ),
    )

    args = parser.parse_args()
    attack_method = args.attack_method
    model_type = args.model_type
    dataset_id = args.dataset_id
    root_data_dir = args.root_data_dir
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    poison_rate = args.poison_rate

# Validate arguments
# ------------------
# The poison rate must be set if attacking, and it must be a positive float lower or equal to 1
if attack_method != "clean" and (
    not poison_rate or poison_rate <= 0 or poison_rate > 1
):
    raise ValueError(
        "When attacking, the poison rate must be a float with value in (0, 1], "
        f"got {poison_rate} instead."
    )


# Surrogate model is only to be trained on the clean dataset
if model_type == "surrogate" and attack_method != "clean":
    raise ValueError(
        "Surrogate models can only be trained on clean training data, stopping."
    )


# Run the training
# -----------------
print(f"{timestamp()} +++ TRAINWRECK STARTED +++", flush=True)
t_start = time()

# Determine the model's transforms
transforms = ImageClassifierFactory.image_classifier_transforms(model_type)

# Create the dataset
t_data = time()
print(f"{timestamp()} Poisoning the data...", flush=True)
dataset = Dataset(dataset_id, root_data_dir, transforms)

# If the attack method is not "clean", poison the dataset with the Trainwreck attack
if attack_method != "clean":
    trainwreck_attack = TrainwreckFactory.trainwreck_attack_obj(
        attack_method, dataset, poison_rate
    )
    trainwreck_attack.attack_dataset()

print(f"{timestamp()} Data poisoned in {t_readable(time() - t_data)}.", flush=True)
# Create the model
model = ImageClassifierFactory.image_classifier_obj(
    model_type, dataset, n_epochs, attack_method, load_existing_model=False
)

# If running on SLURM, the empirical batch size is going to be used.
if slurm:
    batch_size = model.slurm_empirical_batch_size()  # pylint: disable=C0103

# Train the model
model.train(batch_size)

print(
    f"{timestamp()} +++ TRAINWRECK FINISHED ({t_readable(time() - t_start)}) +++",
    flush=True,
)
