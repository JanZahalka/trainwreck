"""
results.py

Result analysis functionality.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

from commons import RESULTS_DIR, RAW_RESULTS_DIR
from dataset.dataset import Dataset
from models.factory import ImageClassifierFactory


class ResultAnalyzer:
    """
    Provides analysis of Trainwreck results.
    """

    RESULT_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
    BEST_METRICS_CSV_PATH = os.path.join(RESULT_ANALYSIS_DIR, "best_metrics.csv")

    @classmethod
    def best_metrics_csv(cls):
        """
        Creates a CSV file reporting on the best values of eval metrics (top-1 acc, top-5 acc,
        CE loss) achieved by the evaluated trained models across the training epochs.
        """
        # Ensure the result analysis dir exists
        cls._create_analysis_dir()

        # Init the CSV string to the header
        best_metrics_csv = "model,best_top1_acc,best_top5_acc,best_loss\n"

        # Iterate over the results
        for raw_result_json_fname in os.listdir(RAW_RESULTS_DIR):
            # Load the raw result
            raw_result_json_path = os.path.join(RAW_RESULTS_DIR, raw_result_json_fname)

            with open(raw_result_json_path, "r", encoding="utf-8") as f:
                raw_result = json.loads(f.read())

            # The model name is simply the file name before ".json"
            model = ".".join(raw_result_json_fname.split(".")[:-1])

            # Initialize the metric trackers
            best_top1_acc = 0.0
            best_top5_acc = 0.0
            best_loss = np.inf

            # Iterate over the results array and record the best vals from epoch 20 onwards
            for epoch_entry in raw_result[19:]:
                best_top1_acc = max(best_top1_acc, epoch_entry["top1"])
                best_top5_acc = max(best_top5_acc, epoch_entry["top5"])
                best_loss = min(best_loss, epoch_entry["loss"])

            # Append the model's entry to the CSV
            best_metrics_csv += f"{model},{best_top1_acc},{best_top5_acc},{best_loss}\n"

        # Save the CSV
        with open(cls.BEST_METRICS_CSV_PATH, "w", encoding="utf-8") as f:
            f.write(best_metrics_csv)

    @classmethod
    def complete_analysis(cls):
        """
        Performs a complete results analysis, running all available analysis methods and
        outputting the results.
        """
        # Outputs a CSV with the best eval metrics achieved by individual models across epochs
        cls.best_metrics_csv()
        cls.poison_rate_plots()

    @classmethod
    def poison_rate_plots(cls):
        """
        Outputs the poison rate plots from poison rate = [0.25, 0.33, 0.5, 0.67, 0.75, 1]. Lines for
        perturbation attacks, points @ 0.25 for swapper attacks
        """
        # Set LaTeX
        # plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

        fig = plt.figure(figsize=(16, 5))
        DATASET_TITLES = ["CIFAR-10", "CIFAR-100"]
        METHOD_COLORS = {
            "trainwreck": "#ff2400",
            "advreplace": "#ccccff",  # Periwinkle
            "randomswap": "#daa520",  # Goldenrod
            "jsdswap": "#b2d63f",  # Chartreuse
        }
        MARKERS = {"efficientnet_v2_s": "s", "resnext_101": "^", "vit_l_16": "p"}
        LABELS_METHODS = {
            "trainwreck": "Trainwreck",
            "advreplace": "AdvReplace",
            "randomswap": "RandomSwap",
            "jsdswap": "JSDSwap",
        }
        LABELS_MODELS = {
            "efficientnet_v2_s": "EfficientNet",
            "resnext_101": "ResNeXt",
            "vit_l_16": "FinetndViT",
        }

        for d, dataset in enumerate(["cifar10", "cifar100"]):
            ax = fig.add_subplot(1, 2, (d + 1))
            ax.set_xlabel("Poison rate (π)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Top-1 test accuracy")
            ax.set_title(DATASET_TITLES[d])

            # Perturbation methods
            for method in ["advreplace", "trainwreck", "randomswap", "jsdswap"]:
                for m, model in enumerate(
                    ["efficientnet_v2_s", "resnext_101", "vit_l_16"]
                ):
                    poison_rates = [
                        0.05,
                        0.1,
                        0.2,
                        0.25,
                        0.33,
                        0.5,
                        0.67,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                        0.95,
                        1.0,
                    ]
                    accuracies = []

                    if method in ["advreplace", "trainwreck"]:
                        eps_str = "-eps8"
                    else:
                        eps_str = ""

                    if method == "trainwreck":
                        method_str = "trainwreck-u"
                    else:
                        method_str = method

                    for pr in poison_rates.copy():
                        results_json_fname = f"{dataset}-{dataset}-{method_str}-pr{pr}{eps_str}-target-{model}-30epochs.json"
                        results_json_path = os.path.join(
                            RAW_RESULTS_DIR, results_json_fname
                        )

                        try:
                            with open(results_json_path, "r", encoding="utf-8") as f:
                                raw_result = json.loads(f.read())
                        except FileNotFoundError:
                            poison_rates.remove(pr)
                            continue

                        best_acc = 0.0

                        for epoch_entry in raw_result[19:]:
                            best_acc = max(best_acc, epoch_entry["top1"])

                        accuracies.append(best_acc / 100)

                    ax.plot(
                        poison_rates,
                        accuracies,
                        color=METHOD_COLORS[method],
                        marker=MARKERS[model],
                        label=f"{LABELS_METHODS[method]}-{LABELS_MODELS[model]}",
                    )

            ax.legend(loc=3, ncols=2, fontsize=9)

        plt.savefig(
            os.path.join(cls.RESULT_ANALYSIS_DIR, "poison_plot.pdf"),
            bbox_inches="tight",
        )

    @classmethod
    def poison_selection(cls):
        """
        Selects a handful of random CIFAR-10 poisoned images for demonstration in the paper.
        """
        img_showcase_dir = os.path.join(cls.RESULT_ANALYSIS_DIR, "img_showcase")

        if not os.path.exists(img_showcase_dir):
            os.makedirs(img_showcase_dir)
        
        # Clean CIFAR-10

        # 20 diverse CIFAR-10 images, clean-TW 1.0
        for i in range(20):


        # Select a random class. 8 TW-1.0 images, 8 RSwap-0.25 images (= 2 swapped)

    @classmethod
    def _create_analysis_dir(cls):
        """
        Creates the results analysis directory if it doesn't exist already.
        """
        if not os.path.exists(cls.RESULT_ANALYSIS_DIR):
            os.makedirs(cls.RESULT_ANALYSIS_DIR)
