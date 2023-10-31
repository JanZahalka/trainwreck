"""
sandbox.py

A place to try code out or perform ad-hoc auxiliary computations.
"""

from timm import utils
import torch
import torchvision

from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.surrogates import SurrogateResNet50

raw_weights = torchvision.models.ResNet50_Weights.DEFAULT
dataset = Dataset("cifar10", "/home/cortex/data", raw_weights.transforms())

_, test_loader = dataset.data_loaders(32, False, 2)

top1 = utils.AverageMeter()
loss_fn = torch.nn.CrossEntropyLoss()


raw_model = torchvision.models.resnet50(weights=raw_weights)
raw_model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
state_dict = torch.load(
    "/home/cortex/sw/trainwreck/models/weights/cifar10-clean-surrogate-resnet50-30epochs.pth"
)
raw_model.load_state_dict(state_dict)
raw_model.eval()
raw_model.cuda()


man_top1_class2 = 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.cuda(), y.cuda()
        output = raw_model(X)
        _, y_pred = torch.max(output, 1)

        loss = loss_fn(output, y)
        top1_acc, top5_acc = utils.accuracy(output, y, topk=(1, 5))
        top1.update(top1_acc.item(), X.size(0))

        man_top1_class2 += sum(torch.logical_and(y == 2, y_pred == y)).item()


print(
    f"Overall top-1: {top1.avg}, class 2 top-1: {man_top1_class2/dataset.n_class_data_items('test', 2)}"
)
