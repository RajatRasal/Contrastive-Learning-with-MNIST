import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim

from .components import LinearHead, MNISTConvEncoder


class MNISTClassifier(pl.LightningModule):
    def __init__(self, activ_type: str, pool_type: str, lr: float):
        super().__init__()
        self.save_hyperparameters()

        # Neural Networks
        self.encoder = MNISTConvEncoder(
            activ_type=self.hparams.activ_type,
            pool_type=self.hparams.pool_type,
        )
        self.head = LinearHead(MNISTConvEncoder.backbone_output_size, 10)
        self.output = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

        # Metrics
        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_loss = torchmetrics.MeanMetric()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_loss = torchmetrics.MeanMetric()
        self.test_acc = torchmetrics.Accuracy()

    def __classifier(self, x):
        return self.output(self.head(self.encoder(x)))

    def forward(self, x):
        return torch.argmax(self.__classifier(x), dim=1)

    def __loss_from_batch(self, batch):
        x, y = batch
        pred = self.__classifier(x)
        return torch.argmax(torch.exp(pred), dim=1), self.loss(pred, y)

    def __step(self, batch, loss_agg, acc_agg):
        pred, loss = self.__loss_from_batch(batch)
        loss_agg.update(loss)
        acc_agg.update(pred, batch[1])
        return loss

    def __epoch_end(self, log_str, loss_agg, acc_agg, prog_bar=False):
        self.log(f"{log_str} Accuracy", acc_agg.compute())
        self.log(f"{log_str} Loss", loss_agg.compute(), prog_bar=prog_bar)
        acc_agg.reset()
        loss_agg.reset()

    def training_step(self, batch, batch_idx):
        return self.__step(batch, self.train_loss, self.train_acc)

    def training_epoch_end(self, outputs):
        self.__epoch_end("Training", self.train_loss, self.train_acc)

    def validation_step(self, batch, batch_idx):
        self.__step(batch, self.valid_loss, self.valid_acc)

    def validation_epoch_end(self, outputs):
        self.__epoch_end("Validation", self.valid_loss, self.valid_acc, True)

    def test_step(self, batch, batch_idx):
        self.__step(batch, self.test_loss, self.test_acc)

    def test_epoch_end(self, outputs):
        self.__epoch_end("Test", self.test_loss, self.test_acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
