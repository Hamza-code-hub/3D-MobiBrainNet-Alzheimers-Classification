"""
3D MRI dataset utilities.

Supported formats:
- .npy
- .nii
- .nii.gz
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


def load_volume(path):
    path = Path(path)

    lower_name = path.name.lower()

    if lower_name.endswith(".npy"):
        volume = np.load(path)

    elif (
        lower_name.endswith(".nii")
        or lower_name.endswith(".nii.gz")
    ):
        volume = nib.load(
            str(path)
        ).get_fdata()

    else:
        raise ValueError(
            f"Unsupported volume: {path}"
        )

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return volume


def normalize_volume(volume):
    mean = float(volume.mean())

    std = float(volume.std())

    if std < 1e-6:
        std = 1.0

    return (
        volume - mean
    ) / std


def resize_volume(
    volume,
    output_size,
):
    tensor = torch.from_numpy(
        volume
    ).float()

    if tensor.ndim != 3:
        raise ValueError(
            "MRI volume must be 3-dimensional."
        )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=output_size,
        mode="trilinear",
        align_corners=False,
    )

    return tensor.squeeze(0)


def discover_dataset(
    root,
    class_names,
):
    root = Path(root)

    samples = []

    for label, class_name in enumerate(
        class_names
    ):
        class_dir = root / class_name

        if not class_dir.exists():
            continue

        files = []

        files.extend(
            class_dir.rglob("*.npy")
        )

        files.extend(
            class_dir.rglob("*.nii")
        )

        files.extend(
            class_dir.rglob("*.nii.gz")
        )

        for path in sorted(set(files)):
            samples.append(
                (
                    str(path),
                    label,
                )
            )

    if not samples:
        raise RuntimeError(
            f"No supported MRI volumes found in {root}"
        )

    return samples


class MRIVolumeDataset(Dataset):

    def __init__(
        self,
        samples,
        volume_size=(96, 96, 96),
        augment=False,
    ):
        self.samples = samples
        self.volume_size = volume_size
        self.augment = augment


    def __len__(self):
        return len(self.samples)


    def _augment(self, x):

        if torch.rand(1).item() < 0.5:
            x = torch.flip(
                x,
                dims=[1],
            )

        if torch.rand(1).item() < 0.5:
            x = torch.flip(
                x,
                dims=[2],
            )

        return x


    def __getitem__(self, index):
        path, label = self.samples[index]

        volume = load_volume(path)

        volume = normalize_volume(
            volume
        )

        volume = resize_volume(
            volume,
            self.volume_size,
        )

        if self.augment:
            volume = self._augment(
                volume
            )

        return {
            "image": volume,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "path": path,
        }
