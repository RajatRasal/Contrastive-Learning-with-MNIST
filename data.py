from pl_bolts.datamodules import MNISTDataModule


def get_datamodule(batch_size: int):
    return MNISTDataModule("~/", batch_size=batch_size)
