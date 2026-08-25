import argparse

import torch

from .data import (
    load_volume,
    normalize_volume,
    resize_volume,
)

from .models import create_model

from .utils import get_device


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--volume",
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

    class_names = checkpoint[
        "class_names"
    ]

    volume_size = tuple(
        checkpoint[
            "volume_size"
        ]
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

    volume = load_volume(
        args.volume
    )

    volume = normalize_volume(
        volume
    )

    volume = resize_volume(
        volume,
        volume_size,
    )

    volume = volume.unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        logits = model(
            volume
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    index = int(
        probabilities.argmax()
    )

    print(
        "\nPrediction"
    )

    print(
        "-" * 40
    )

    print(
        f"Class      : "
        f"{class_names[index]}"
    )

    print(
        f"Confidence : "
        f"{probabilities[index].item():.2%}"
    )

    print(
        "\nProbabilities"
    )

    print(
        "-" * 40
    )

    for name, probability in zip(
        class_names,
        probabilities,
    ):
        print(
            f"{name:<12}: "
            f"{probability.item():.2%}"
        )


if __name__ == "__main__":
    main()
