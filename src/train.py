import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    LEARNING_RATE,
    MODEL_DIR,
    RANDOM_SEED,
    VOLUME_SIZE,
    WEIGHT_DECAY,
)

from .data import (
    MRIVolumeDataset,
    discover_dataset,
)

from .models import (
    create_model,
    parameter_count,
)

from .utils import (
    get_device,
    save_json,
    set_seed,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train 3D Alzheimer's "
            "classification model"
        )
    )

    parser.add_argument(
        "--data",
        default="dataset",
    )

    parser.add_argument(
        "--model",
        default="mobibrainnet",
        choices=[
            "mobibrainnet",
            "cnn3d",
            "resnet3d",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    return parser.parse_args()


def accuracy(
    logits,
    labels,
):
    predictions = logits.argmax(
        dim=1
    )

    return (
        predictions == labels
    ).float().mean().item()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch in loader:

        images = batch["image"].to(
            device
        )

        labels = batch["label"].to(
            device
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            if training:
                loss.backward()

                optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += (
            logits.argmax(1)
            == labels
        ).sum().item()

        total_samples += batch_size

    return {
        "loss": (
            total_loss
            / max(
                total_samples,
                1,
            )
        ),
        "accuracy": (
            total_correct
            / max(
                total_samples,
                1,
            )
        ),
    }


def main():
    args = parse_args()

    set_seed(
        RANDOM_SEED
    )

    device = get_device()

    samples = discover_dataset(
        args.data,
        CLASS_NAMES,
    )

    paths = [
        item[0]
        for item in samples
    ]

    labels = [
        item[1]
        for item in samples
    ]

    train_paths, val_paths, \
    train_labels, val_labels = \
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

    train_samples = list(
        zip(
            train_paths,
            train_labels,
        )
    )

    val_samples = list(
        zip(
            val_paths,
            val_labels,
        )
    )

    train_dataset = MRIVolumeDataset(
        train_samples,
        VOLUME_SIZE,
        augment=True,
    )

    val_dataset = MRIVolumeDataset(
        val_samples,
        VOLUME_SIZE,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = create_model(
        args.model,
        num_classes=len(
            CLASS_NAMES
        ),
    ).to(device)

    print(
        f"Device: {device}"
    )

    print(
        "Trainable parameters:",
        f"{parameter_count(model):,}",
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        MODEL_DIR
        / f"{args.model}_best.pt"
    )

    history = []

    best_loss = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        record = {
            "epoch": epoch,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_accuracy": (
                train_metrics[
                    "accuracy"
                ]
            ),
            "val_loss": (
                val_metrics["loss"]
            ),
            "val_accuracy": (
                val_metrics[
                    "accuracy"
                ]
            ),
        }

        history.append(
            record
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss "
            f"{record['train_loss']:.4f} | "
            f"train acc "
            f"{record['train_accuracy']:.4f} | "
            f"val loss "
            f"{record['val_loss']:.4f} | "
            f"val acc "
            f"{record['val_accuracy']:.4f}"
        )

        if (
            val_metrics["loss"]
            < best_loss
        ):
            best_loss = (
                val_metrics["loss"]
            )

            torch.save(
                {
                    "model_name": args.model,
                    "class_names": CLASS_NAMES,
                    "volume_size": VOLUME_SIZE,
                    "state_dict": (
                        model.state_dict()
                    ),
                },
                checkpoint,
            )

    save_json(
        history,
        "outputs/training_history.json",
    )

    print(
        f"\nBest checkpoint: "
        f"{checkpoint}"
    )


if __name__ == "__main__":
    main()
