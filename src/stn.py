"""
Spatial Transformer Network for MNIST images. Architecture taken from
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTSpatialTransformer(nn.Module):
    def __init__(self, localization_latent_dim: int = 16):
        super().__init__()

        self.localization_latent_dim = localization_latent_dim

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, self.localization_latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.localization_latent_dim, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
