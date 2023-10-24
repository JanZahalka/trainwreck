"""
sandbox.py

A place to try code out or perform ad-hoc auxiliary computations.
"""
import copy

from datasets.dataset import Dataset
from datasets.poisoners import PoisonerFactory

i1 = 2612
i2 = 24788

for dataset_id in Dataset.valid_dataset_ids():
    dataset = Dataset(dataset_id, "/home/cortex/data", None)
    poisoner = PoisonerFactory.poisoner_obj(dataset)

    X1_old, y1_old = copy.deepcopy(dataset.train_dataset[i1])
    X2_old, y2_old = copy.deepcopy(dataset.train_dataset[i2])

    poisoner.swap_item_data(i1, i2)

    X1_new, y1_new = dataset.train_dataset[i1]
    X2_new, y2_new = dataset.train_dataset[i2]

    assert X1_new == X2_old
    assert y1_new == y1_old
    assert X2_new == X1_old
    assert y2_new == y2_old

print("Diagnostics check OK.")

# train_dataset.set_image_data(2407, "ludekbina")
# train_dataset.set_image_label(2407, "ludekbina")
