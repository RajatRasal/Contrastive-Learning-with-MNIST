from pl_bolts.datamodules import MNISTDataModule


def get_datamodule(batch_size: int, normalize: bool = False):
    return MNISTDataModule("~/", batch_size=batch_size, normalize=normalize)
