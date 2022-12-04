import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch import nn


class MNISTSupContrast(pl.LightningModule):

    def __init__(self, encoder: nn.Module, head: nn.Module, lr: float):
        super().__init__()
        self.lr = lr

        self.encoder = nn.Sequential(encoder, head)
        self.loss = losses.ContrastiveLoss(pos_margin=0.25, neg_margin=1.5)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.encoder(x)
        loss = self.loss(embeddings, y)
        if batch_idx % 10 == 0:
            self.log("Training Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.encoder(x)
        loss = self.loss(embeddings, y)
        self.log("Validation Loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
