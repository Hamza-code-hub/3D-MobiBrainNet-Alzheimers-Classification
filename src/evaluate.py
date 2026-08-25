import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from .config import (
    CLASS_NAMES,
    RANDOM_SEED,
)

from .data import (
    MRIVolumeDataset,
    discover_dataset,
)

from .models import create_model

from .utils import get_device


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="dataset",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )

    volume_size = tuple(
        checkpoint.get(
            "volume_size",
            (96, 96, 96),
        )
    )

    model = create_model(
        checkpoint["model_name"],
        len(class_names),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(device)

    model.eval()

    samples = discover_dataset(
        args.data,
        class_names,
    )

    paths = [
        sample[0]
        for sample in samples
    ]

    labels = [
        sample[1]
        for sample in samples
    ]

    _, val_paths, _, val_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    dataset = MRIVolumeDataset(
        list(
            zip(
                val_paths,
                val_labels,
            )
        ),
        volume_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    targets = []

    predictions = []

    probabilities = []

    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(device)

            labels_batch = batch[
                "label"
            ].to(device)

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = probs.argmax(
                dim=1
            )

            targets.extend(
                labels_batch.cpu().numpy()
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    targets = np.asarray(
        targets
    )

    predictions = np.asarray(
        predictions
    )

    probabilities = np.asarray(
        probabilities
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy
        ),
        "precision_macro": float(
            precision
        ),
        "recall_macro": float(
            recall
        ),
        "f1_macro": float(
            f1
        ),
    }

    try:
        one_hot = np.eye(
            len(class_names)
        )[targets]

        metrics["roc_auc_macro"] = (
            float(
                roc_auc_score(
                    one_hot,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
        )

    except ValueError:
        pass

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    with open(
        output_dir
        / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            report
        )

    matrix = confusion_matrix(
        targets,
        predictions,
    )

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    display = ConfusionMatrixDisplay(
        matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    one_hot = np.eye(
        len(class_names)
    )[targets]

    figure = plt.figure(
        figsize=(8, 6)
    )

    for index, class_name in enumerate(
        class_names
    ):

        try:
            fpr, tpr, _ = roc_curve(
                one_hot[:, index],
                probabilities[:, index],
            )

            plt.plot(
                fpr,
                tpr,
                label=class_name,
            )

        except ValueError:
            continue

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "One-vs-Rest ROC Curves"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        output_dir
        / "roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()

    print(
        report
    )


if __name__ == "__main__":
    main()
