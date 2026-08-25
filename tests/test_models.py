import torch

from src.models import (
    BaselineCNN3D,
    MobiBrainNet3D,
    ResNet3D,
)


INPUT = torch.randn(
    1,
    1,
    64,
    64,
    64,
)


def test_mobibrainnet():
    model = MobiBrainNet3D(
        num_classes=3
    )

    output = model(INPUT)

    assert output.shape == (
        1,
        3,
    )


def test_cnn3d():
    model = BaselineCNN3D(
        num_classes=3
    )

    output = model(INPUT)

    assert output.shape == (
        1,
        3,
    )


def test_resnet3d():
    model = ResNet3D(
        num_classes=3
    )

    output = model(INPUT)

    assert output.shape == (
        1,
        3,
    )
