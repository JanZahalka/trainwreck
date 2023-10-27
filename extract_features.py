"""
extract_features.py

Extracts features used by the Trainwreck attack from the datasets.
"""
import argparse

from datasets.dataset import Dataset
from models.featextr import ImageNetViTL16FeatureExtractor

parser = argparse.ArgumentParser()
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

args = parser.parse_args()


for dataset_id in ["cifar10"]:
    dataset = Dataset(
        dataset_id, args.root_data_dir, ImageNetViTL16FeatureExtractor.transforms()
    )
    feat_extractor = ImageNetViTL16FeatureExtractor()
    feat_extractor.extract_features(dataset, args.batch_size)
