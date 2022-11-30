import argparse
import os

import torch

from classify import MNISTClassifier
from contrastive import MNISTSupContrast
from data import get_datamodule
from plot import pca_proj, tsne_proj
from train import get_trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--contrastive", action="store_true")
    args = parser.parse_args()

    dm = get_datamodule(256)

    if args.contrastive:
        path = "/Users/work/Documents/contrastive_learning_mnist/lightning_logs/version_105"  # noqa: E501
        model = MNISTSupContrast.load_from_checkpoint(
            os.path.join(path, "checkpoints", "last.ckpt")
        )
    else:
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
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels)

    pca_proj(embeddings, labels)
    tsne_proj(embeddings, labels)
