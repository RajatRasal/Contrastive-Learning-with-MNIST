import os

import torch

from classify import MNISTClassifier
from data import get_datamodule
from plot import pca_proj
from train import get_trainer

if __name__ == "__main__":
    dm = get_datamodule(256)

    path = "/Users/work/Documents/contrastive_learning_mnist/lightning_logs/version_101"  # noqa: E501
    model = MNISTClassifier.load_from_checkpoint(
        os.path.join(path, "checkpoints", "last.ckpt")
    )

    trainer = get_trainer(None, None, logger=False)
    trainer.test(model, datamodule=dm)

    model.freeze()

    embeddings, labels = [], []
    for data, label in dm.test_dataloader():
        embeddings.append(model.encoder(data))
        labels.append(label)

    pca_proj(torch.cat(embeddings, dim=0), torch.cat(labels))
