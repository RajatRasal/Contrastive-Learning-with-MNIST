import argparse

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from classify import MNISTClassifier
from components import LinearHead, MNISTConvEncoder
from contrastive import MNISTSupContrast
from data import get_datamodule


def get_trainer(max_epochs: int, val_check_freq: int, callbacks=None):
    return pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=val_check_freq,
        callbacks=callbacks,
    )


def train_mnist_classifier(trainer, batch_size, lr, pooling, activation, seed):
    seed_everything(seed)
    dm = get_datamodule(batch_size)
    model = MNISTClassifier(activation, pooling, lr)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrain", action="store_true")
    # TODO: Change to choice between keys in _ACTIVATIONS constant
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-p", "--pooling", type=str, default="max")
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("-l", "--lr", type=float, default=0.07)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-v", "--val-check-freq", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=1234)
    args = parser.parse_args()

    if not args.pretrain:
        trainer = get_trainer(
            max_epochs=args.epochs,
            val_check_freq=args.val_check_freq,
            callbacks=[
                ModelCheckpoint(
                    save_last=True,
                    monitor="Validation Loss",
                    filename="epoch={epoch:02d}-loss={Validation Loss:.5f}-acc={Validation Accuracy:.5f}",  # noqa: E501
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
        seed_everything(args.seed)

        dm = get_datamodule(256)
        encoder = MNISTConvEncoder()
        sup_con_head = LinearHead(MNISTConvEncoder.backbone_output_size, 256)
        model = MNISTSupContrast(encoder, sup_con_head, lr=5e-3)
        get_trainer(max_epochs=10, val_check_freq=5).fit(model, datamodule=dm)

        dm = get_datamodule(256)
        cls_head = LinearHead(MNISTConvEncoder.backbone_output_size, 10)
        model = MNISTClassifier(
            encoder.requires_grad_(False), cls_head, lr=1e-3
        )
        trainer = get_trainer(max_epochs=10, val_check_freq=5)
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
