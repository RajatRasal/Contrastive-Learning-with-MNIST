import kornia
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_metric_learning import losses

from .components import LinearHead, MNISTConvEncoder


class MNISTSupContrast(pl.LightningModule):
    def __init__(
        self,
        activ_type: str,
        pool_type: str,
        head_output: int,
        lr: float,
        pos_margin: float = 0.25,
        neg_margin: float = 1.5,
        preprocess: bool = False,
        dropout: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Preprocessing
        self.preprocessing = kornia.augmentation.RandomAffine(
            # degrees=0,
            degrees=(-40, 40),
            translate=0.25,
            scale=[0.5, 1.5],
            shear=45,
        )

        # Neural Networks
        self.encoder = MNISTConvEncoder(
            activ_type=self.hparams.activ_type,
            pool_type=self.hparams.pool_type,
        )
        self.head = LinearHead(
            MNISTConvEncoder.backbone_output_size,
            self.hparams.head_output,
            self.hparams.dropout,
        )
        self.loss = losses.ContrastiveLoss(
            pos_margin=self.hparams.pos_margin,
            neg_margin=self.hparams.neg_margin,
        )

        # Metrics
        self.train_loss = torchmetrics.MeanMetric()
        # self.train_acc = torchmetrics.Accuracy()
        self.valid_loss = torchmetrics.MeanMetric()
        # self.valid_acc = torchmetrics.Accuracy()
        self.test_loss = torchmetrics.MeanMetric()
        # self.test_acc = torchmetrics.Accuracy()

    def __step(self, batch, loss_agg, test=False):
        x, y = batch
        if self.hparams.preprocess and not test:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        loss_agg.update(loss)
        return loss

    def __epoch_end(self, log_str, loss_agg, prog_bar=False):
        self.log(f"{log_str} Loss", loss_agg.compute(), prog_bar=prog_bar)
        loss_agg.reset()

    def forward(self, x: torch.Tensor):
        return self.head(self.encoder(x))

    def training_step(self, batch, batch_idx):
        return self.__step(batch, self.train_loss)

    def training_epoch_end(self, outputs):
        self.__epoch_end("Training", self.train_loss)

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, self.valid_loss)

    def validation_epoch_end(self, outputs):
        self.__epoch_end("Validation", self.valid_loss)

    def test_step(self, batch, batch_idx):
        return self.__step(batch, self.test_loss, test=True)

    def test_epoch_end(self, outputs):
        self.__epoch_end("Test", self.test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
