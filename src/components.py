import torch
import torch.nn.functional as F
from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

_POOLING = {
    "max": nn.MaxPool2d,
    "avg": nn.AvgPool2d,
}


class ConvUnit(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_type: str,
        pool_kernel_size: int,
        pool_stride: int,
        activ_type: str,
        **activ_kwargs,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.activ = _ACTIVATIONS[activ_type](**activ_kwargs)
        self.pool = _POOLING[pool_type](
            kernel_size=pool_kernel_size, stride=pool_stride
        )

    def forward(self, x):
        return self.pool(self.activ(self.bn(self.conv(x))))


class MNISTConvEncoder(nn.Module):
    backbone_output_size = 196

    def __init__(self, activ_type: str, pool_type: str):
        super().__init__()

        self.conv_unit1 = ConvUnit(
            input_channels=1,
            output_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            pool_type=pool_type,
            pool_kernel_size=2,
            pool_stride=2,
            activ_type=activ_type,
        )

        self.conv_unit2 = ConvUnit(
            input_channels=4,
            output_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            pool_type=pool_type,
            pool_kernel_size=2,
            pool_stride=2,
            activ_type=activ_type,
        )

    def forward(self, x: torch.Tensor):
        out1 = self.conv_unit1(x)
        out2 = self.conv_unit2(out1)
        return out2.view(-1, self.backbone_output_size)


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0):
        super().__init__()
        self.dropout = dropout
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.head(x)
        if self.dropout:
            out = F.dropout(out, p=self.dropout)
        return out
