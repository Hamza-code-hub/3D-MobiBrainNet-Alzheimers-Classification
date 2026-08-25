"""
3D neural-network architectures.

Models
-------
MobiBrainNet3D
BaselineCNN3D
ResNet3D
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
    ):
        padding = kernel_size // 2

        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )


class Mobile3DBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )

        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )


    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class MobiBrainNet3D(nn.Module):

    def __init__(
        self,
        num_classes=3,
        dropout=0.30,
    ):
        super().__init__()

        self.features = nn.Sequential(

            ConvBNAct(
                1,
                16,
                stride=2,
            ),

            Mobile3DBlock(
                16,
                32,
                stride=1,
            ),

            Mobile3DBlock(
                32,
                64,
                stride=2,
            ),

            Mobile3DBlock(
                64,
                96,
                stride=2,
            ),

            Mobile3DBlock(
                96,
                160,
                stride=2,
            ),

            Mobile3DBlock(
                160,
                256,
                stride=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool3d(
            1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes,
            ),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class BaselineCNN3D(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):
        super().__init__()

        self.features = nn.Sequential(

            ConvBNAct(
                1,
                32,
            ),

            nn.MaxPool3d(2),

            ConvBNAct(
                32,
                64,
            ),

            nn.MaxPool3d(2),

            ConvBNAct(
                64,
                128,
            ),

            nn.MaxPool3d(2),

            ConvBNAct(
                128,
                256,
            ),
        )

        self.pool = nn.AdaptiveAvgPool3d(
            1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(
                256,
                num_classes,
            ),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class Residual3DBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            stride=stride,
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(
                out_channels
            ),
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.skip = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(
                    out_channels
                ),
            )

        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(
            inplace=True
        )


    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = x + identity

        return self.relu(x)


class ResNet3D(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):
        super().__init__()

        self.stem = ConvBNAct(
            1,
            32,
            kernel_size=7,
            stride=2,
        )

        self.layers = nn.Sequential(

            Residual3DBlock(
                32,
                32,
            ),

            Residual3DBlock(
                32,
                64,
                stride=2,
            ),

            Residual3DBlock(
                64,
                128,
                stride=2,
            ),

            Residual3DBlock(
                128,
                256,
                stride=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool3d(
            1
        )

        self.fc = nn.Linear(
            256,
            num_classes,
        )


    def forward(self, x):
        x = self.stem(x)

        x = self.layers(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return self.fc(x)


def create_model(
    model_name,
    num_classes=3,
):
    name = model_name.lower()

    if name in {
        "mobibrainnet",
        "mobi",
    }:
        return MobiBrainNet3D(
            num_classes
        )

    if name in {
        "cnn3d",
        "3dcnn",
        "cnn",
    }:
        return BaselineCNN3D(
            num_classes
        )

    if name in {
        "resnet3d",
        "resnet",
    }:
        return ResNet3D(
            num_classes
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
