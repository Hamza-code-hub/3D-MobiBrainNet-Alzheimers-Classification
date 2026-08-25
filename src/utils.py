import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def save_json(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )
