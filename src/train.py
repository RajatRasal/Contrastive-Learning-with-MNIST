import argparse

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from .classify import MNISTClassifier
from .contrastive import MNISTSupContrast
from .data import get_datamodule


def get_trainer(
    max_epochs: int, val_check_freq: int, callbacks=None, logger=True
):
    return pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=val_check_freq,
        callbacks=callbacks,
        logger=logger,
    )


def train_mnist_classifier(trainer, batch_size, lr, pooling, activation, seed):
    seed_everything(seed)
    dm = get_datamodule(batch_size)
    model = MNISTClassifier(activation, pooling, lr)
    trainer.fit(model, datamodule=dm)


def train_mnist_contrastive(
    trainer,
    batch_size,
    no_normalise,
    embedding,
    lr,
    pooling,
    activ,
    seed,
    pos_margin: float = 0.25,
    neg_margin: float = 1.5,
    preprocess: bool = False,
    dropout: float = 0.5,
    stn: bool = False,
    stn_latent_dim: int = 32,
):
    seed_everything(seed)
    dm = get_datamodule(batch_size, no_normalise)
    model = MNISTSupContrast(
        activ,
        pooling,
        embedding,
        lr,
        pos_margin,
        neg_margin,
        preprocess,
        dropout,
        stn,
        stn_latent_dim,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrain", action="store_true")
    # TODO: Change to choice between keys in _ACTIVATIONS constant
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-p", "--pooling", type=str, default="max")
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("-d", "--embedding", type=int, default=256)
    parser.add_argument("-l", "--lr", type=float, default=0.07)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-v", "--val-check-freq", type=int, default=5)
    parser.add_argument("-pm", "--pos-margin", type=float, default=1.5)
    parser.add_argument("-nm", "--neg-margin", type=float, default=0.5)
    parser.add_argument(
        "-pr", "--preprocess", default=None, choices=["RandAffine", "RandAug"]
    )
    parser.add_argument("-dr", "--dropout", type=float, default=0)
    parser.add_argument("-st", "--stn", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=1234)
    parser.add_argument("-n", "--no-normalise", action="store_true")
    parser.add_argument("-stdim", "--stn-latent-dim", type=int, default=32)
    args = parser.parse_args()

    if not args.pretrain:
        trainer = get_trainer(
            max_epochs=args.epochs,
            val_check_freq=args.val_check_freq,
            callbacks=[
                ModelCheckpoint(
                    save_last=True,
                    monitor="Validation Loss",
                    filename="epoch={epoch:02d}-loss={Validation Loss:.9f}-acc={Validation Accuracy:.9f}",  # noqa: E501
                ),
            ],
        )
        train_mnist_classifier(
            trainer,
            args.batch_size,
            args.lr,
            args.pooling,
            args.activation,
            args.seed,
        )
    else:
        trainer = get_trainer(
            max_epochs=args.epochs,
            val_check_freq=args.val_check_freq,
            callbacks=[
                ModelCheckpoint(
                    save_last=True,
                    monitor="Validation Loss",
                    filename="epoch={epoch:02d}-loss={Validation Loss:.9f}",
                ),
            ],
        )
        train_mnist_contrastive(
            trainer,
            args.batch_size,
            args.no_normalise,
            args.embedding,
            args.lr,
            args.pooling,
            args.activation,
            args.seed,
            args.pos_margin,
            args.neg_margin,
            args.preprocess,
            args.dropout,
            args.stn,
            args.stn_latent_dim,
        )
