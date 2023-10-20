"""
train_model.py

Trains a surrogate or a target model.
"""
import argparse
import itertools
import os
import sys

from commons import timestamp
from datasets.dataset import Dataset, valid_dataset_ids
from models.surrogates import SurrogateResNet50
from models.targets import EfficientNetV2S, ResNeXt101, FinetunedViTL16

DEFAULT_N_EPOCHS = 30
MODEL_TYPES = ["surrogate", "efficientnet", "resnext", "vit"]

# NOTE: the SLURM* constants are applicable ONLY for the cluster
# The data root dir on the cluster
SLURM_ROOT_DATA_DIR = "/home/zahalja1/data"


# Determine whether the script is running on a SLURM cluster or from command line
try:
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    slurm_array_size = int(os.getenv("SLURM_ARRAY_TASK_MAX")) + 1
    running_on_slurm = True  # pylint: disable=C0103
except TypeError:
    running_on_slurm = False  # pylint: disable=C0103

# If running on a SLURM cluster, autofill the parameters based on the job array ID and size
if running_on_slurm:
    # Enumerate all training parameters in a list of lists
    train_config_params = [valid_dataset_ids(), MODEL_TYPES]

    # Perform a cartesian product of the list of lists to enumerate all individual training sessions
    train_config_all_combinations = list(itertools.product(*train_config_params))
    n_train_combinations = len(train_config_all_combinations)

    # If the task ID is greater than the number of combinations, stop. The training is fully
    # handled by the other processes.
    if slurm_array_task_id >= n_train_combinations:
        print(
            f"{timestamp()} The SLURM job ID is {slurm_array_task_id}, "
            f"but there are only {n_train_combinations} train config combinations. Stopping."
        )
        sys.exit()

    train_config = train_config_all_combinations[slurm_array_task_id]
    dataset_id = train_config[0]
    model_type = train_config[1]
    root_data_dir = SLURM_ROOT_DATA_DIR  # pylint: disable=C0103
    # batch size to be set to the empirical value from the model class later
    n_epochs = DEFAULT_N_EPOCHS  # pylint: disable=C0103


# If running from the command line, parse command line args
else:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        type=str,
        choices=MODEL_TYPES,
        help="The type of model to be trained (surrogate or one of the target models)",
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        choices=["cifar10", "cifar100", "gtsrb"],
        help="The dataset to finetune on.",
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

    args = parser.parse_args()
    model_type = args.model_type
    dataset_id = args.dataset_id
    root_data_dir = args.root_data_dir
    batch_size = args.batch_size
    n_epochs = args.n_epochs

# Run the training
# -----------------
# Determine the model class
if model_type == "surrogate":
    ModelClass = SurrogateResNet50
elif model_type == "efficientnet":
    ModelClass = EfficientNetV2S
elif model_type == "resnext":
    ModelClass = ResNeXt101
elif model_type == "vit":
    ModelClass = FinetunedViTL16
else:
    # This shouldn't happen, should be covered by argparse choices/hardcoded in the SLURM branch,
    # but just to be sure...
    raise ValueError(f"Invalid model type '{model_type}'.")

# If running on SLURM, the empirical batch size is going to be used.
if running_on_slurm:
    batch_size = ModelClass.slurm_empirical_batch_size()  # pylint: disable=C0103

transforms = ModelClass.transforms()
dataset = Dataset(dataset_id, root_data_dir, transforms)
model = ModelClass(dataset, n_epochs, load_existing_model=False)
model.train(batch_size)
