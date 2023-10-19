"""
sandbox.py
"""
import copy
import torchvision


class PoisonedCIFAR10(torchvision.datasets.CIFAR10):
    def set_image_data(self, i, new_image):
        self.data[i] = new_image

    def set_image_label(self, i, new_label):
        self.targets[i] = new_label

    def swap_items(self, i1, i2):
        i1_old_data = copy.deepcopy(self.data[i1])
        i1_old_label = copy.deepcopy(self.targets[i1])

        self.data[i1] = self.data[i2]
        self.targets[i1] = self.targets[i2]

        self.data[i2] = i1_old_data
        self.targets[i2] = i1_old_label


train_dataset = PoisonedCIFAR10(root="/home/cortex/data", train=True, download=False)
test_dataset = PoisonedCIFAR10(root="/home/cortex/data", train=False, download=False)

old_2407_data = copy.deepcopy(train_dataset[2407][0])
old_2407_labels = copy.deepcopy(train_dataset[2407][1])
old_703_data = copy.deepcopy(train_dataset[703][0])
old_703_labels = copy.deepcopy(train_dataset[703][1])

train_dataset.swap_items(2407, 703)

assert train_dataset[2407][0] == old_703_data
assert train_dataset[2407][1] == old_703_labels
assert train_dataset[703][0] == old_2407_data
assert train_dataset[703][1] == old_2407_labels
print("Yeaow!")
