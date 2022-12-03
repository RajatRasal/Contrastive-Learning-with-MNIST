import kornia
import matplotlib.pyplot as plt
from lightning_lite.utilities.seed import seed_everything

from data import get_datamodule


def main():
    seed_everything(1234)

    rotate = kornia.augmentation.RandomAffine(degrees=30)
    transform = kornia.augmentation.RandomAffine(degrees=0, translate=0.25)
    scale = kornia.augmentation.RandomAffine(degrees=0, scale=[0.5, 1.5])
    shear = kornia.augmentation.RandomAffine(degrees=0, shear=45)
    combined = kornia.augmentation.RandomAffine(
        degrees=30, translate=0.25, scale=[0.5, 1.5], shear=45
    )

    nums = 10
    _, ax = plt.subplots(nrows=nums, ncols=6)
    ax[0][0].set_title("original")
    ax[0][1].set_title("rotate")
    ax[0][2].set_title("transform")
    ax[0][3].set_title("scale")
    ax[0][4].set_title("shear")
    ax[0][5].set_title("combined")

    dm = get_datamodule(nums)
    dm.setup()

    for (x, y) in dm.train_dataloader():
        for i, (img, _) in enumerate(zip(x, y)):
            ax[i][0].imshow(img.squeeze(0).numpy(), cmap="gray")
            ax[i][1].imshow(
                rotate(img).squeeze(0).squeeze(0).numpy(), cmap="gray"
            )
            ax[i][2].imshow(
                transform(img).squeeze(0).squeeze(0).numpy(), cmap="gray"
            )
            ax[i][3].imshow(
                scale(img).squeeze(0).squeeze(0).numpy(), cmap="gray"
            )
            ax[i][4].imshow(
                shear(img).squeeze(0).squeeze(0).numpy(), cmap="gray"
            )
            ax[i][5].imshow(
                combined(img).squeeze(0).squeeze(0).numpy(), cmap="gray"
            )
        break

    plt.show()


if __name__ == "__main__":
    main()
