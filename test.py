import os

from classify import MNISTClassifier
from data import get_datamodule
from train import get_trainer

if __name__ == "__main__":
    dm = get_datamodule(256)

    path = "/Users/work/Documents/contrastive_learning_mnist/lightning_logs/version_80"  # noqa: E501
    model = MNISTClassifier.load_from_checkpoint(
        os.path.join(path, "checkpoints", "last.ckpt")
    )

    trainer = get_trainer(None, None)
    trainer.test(model, datamodule=dm)
