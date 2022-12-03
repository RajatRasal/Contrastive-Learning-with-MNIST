`conda activate contrastive_learning_mnist`
`conda env update -f environment.yml`

`tensorboard --logdir lightning_logs`


#### Without pretraining
Fixed hyperparameters:
- batch size = 256
- epochs = 20

| Activation   | Pooling   |   LR |   Val Accuracy |   Train Accuracy |
|--------------|-----------|------|-----------|-------------|
| relu         | max       | 0.01 |  0.974917 |    0.981208 |
| gelu         | max       | 0.01 |  0.978167 |    0.982437 |
| relu         | max       | 0.03 |  0.975833 |    0.979625 |
| gelu         | max       | 0.03 |  0.978083 |    0.982375 |
| relu         | max       | 0.05 |  0.973417 |    0.979417 |
| gelu         | max       | 0.05 |  0.973    |    0.979042 |
| relu         | max       | 0.07 |  0.968833 |    0.969938 |
| gelu         | max       | 0.07 |  0.970583 |    0.977521 |
| relu         | max       | 0.09 |  0.95375  |    0.96275  |
| gelu         | max       | 0.09 |  0.967583 |    0.975812 |
| relu         | avg       | 0.01 |  0.977583 |    0.985771 |
| gelu         | avg       | 0.01 |  0.968417 |    0.98625  |
| relu         | avg       | 0.03 |  0.976167 |    0.984542 |
| gelu         | avg       | 0.03 |  0.97175  |    0.985812 |
| relu         | avg       | 0.05 |  0.979333 |    0.984187 |
| **gelu**         | **avg**       | **0.05** |  **0.98025**  |    **0.985104** |
| relu         | avg       | 0.07 |  0.97425  |    0.980271 |
| gelu         | avg       | 0.07 |  0.976417 |    0.982313 |
| relu         | avg       | 0.09 |  0.973583 |    0.976896 |
| gelu         | avg       | 0.09 |  0.977583 |    0.978625 |

#### Pretraining
ReLU better than GeLU when everything else is fixed - small batch sizes.
For larger batch sizes, GeLU is better.
Small learning rates.

+--------------------------------------+------------+-----------------+--------------+-----------+--------+--------------+------------+
| Trial name                           | status     | loc             | activation   | pooling   |     lr |   batch_size |   val_loss |
|--------------------------------------+------------+-----------------+--------------+-----------+--------+--------------+------------|
| _train_mnist_contrastive_491a6_00000 | TERMINATED | 127.0.0.1:1691  | relu         | max       | 0.01   |         1024 |   0.558239 |
| _train_mnist_contrastive_491a6_00001 | TERMINATED | 127.0.0.1:1709  | gelu         | max       | 0.01   |         1024 |   0.540568 |
| _train_mnist_contrastive_491a6_00002 | TERMINATED | 127.0.0.1:1710  | relu         | max       | 0.01   |         2048 |   0.528964 |
| _train_mnist_contrastive_491a6_00003 | TERMINATED | 127.0.0.1:3441  | gelu         | max       | 0.01   |         2048 |   0.522437 |
| _train_mnist_contrastive_491a6_00004 | TERMINATED | 127.0.0.1:3491  | relu         | max       | 0.001  |         1024 |   0.516201 |
| _train_mnist_contrastive_491a6_00005 | TERMINATED | 127.0.0.1:4723  | gelu         | max       | 0.001  |         1024 |   0.513625 |
| _train_mnist_contrastive_491a6_00006 | TERMINATED | 127.0.0.1:6146  | relu         | max       | 0.001  |         2048 |   0.527985 |
| _train_mnist_contrastive_491a6_00007 | TERMINATED | 127.0.0.1:7356  | gelu         | max       | 0.001  |         2048 |   0.52203  |
| _train_mnist_contrastive_491a6_00008 | TERMINATED | 127.0.0.1:7377  | relu         | max       | 0.0001 |         1024 |   0.624486 |
| _train_mnist_contrastive_491a6_00009 | TERMINATED | 127.0.0.1:15497 | gelu         | max       | 0.0001 |         1024 |   0.62843  |
| _train_mnist_contrastive_491a6_00010 | TERMINATED | 127.0.0.1:15518 | relu         | max       | 0.0001 |         2048 |   0.72802  |
| _train_mnist_contrastive_491a6_00011 | TERMINATED | 127.0.0.1:16800 | gelu         | max       | 0.0001 |         2048 |   0.726901 |
| _train_mnist_contrastive_491a6_00012 | TERMINATED | 127.0.0.1:17264 | relu         | avg       | 0.01   |         1024 |   0.594014 |
| _train_mnist_contrastive_491a6_00013 | TERMINATED | 127.0.0.1:19390 | gelu         | avg       | 0.01   |         1024 |   0.496038 |
| _train_mnist_contrastive_491a6_00014 | TERMINATED | 127.0.0.1:19766 | relu         | avg       | 0.01   |         2048 |   0.870534 |
| _train_mnist_contrastive_491a6_00015 | TERMINATED | 127.0.0.1:25174 | gelu         | avg       | 0.01   |         2048 |   0.482411 |
| _train_mnist_contrastive_491a6_00016 | TERMINATED | 127.0.0.1:25579 | relu         | avg       | 0.001  |         1024 |   0.486609 |
| _train_mnist_contrastive_491a6_00017 | TERMINATED | 127.0.0.1:27185 | gelu         | avg       | 0.001  |         1024 |   0.492692 |
| _train_mnist_contrastive_491a6_00018 | TERMINATED | 127.0.0.1:27227 | relu         | avg       | 0.001  |         2048 |   0.497761 |
| _train_mnist_contrastive_491a6_00019 | TERMINATED | 127.0.0.1:28067 | gelu         | avg       | 0.001  |         2048 |   0.49835  |
| _train_mnist_contrastive_491a6_00020 | TERMINATED | 127.0.0.1:28899 | relu         | avg       | 0.0001 |         1024 |   0.591419 |
| _train_mnist_contrastive_491a6_00021 | TERMINATED | 127.0.0.1:30546 | gelu         | avg       | 0.0001 |         1024 |   0.588001 |
| _train_mnist_contrastive_491a6_00022 | TERMINATED | 127.0.0.1:30964 | relu         | avg       | 0.0001 |         2048 |   0.686366 |
| _train_mnist_contrastive_491a6_00023 | TERMINATED | 127.0.0.1:31399 | gelu         | avg       | 0.0001 |         2048 |   0.687941 |


+--------------------------------------+------------+-----------------+--------------+--------------+-------------+--------------+------------+
| Trial name                           | status     | loc             |   neg_margin |   pos_margin |   embedding |   train_loss |   val_loss |
|--------------------------------------+------------+-----------------+--------------+--------------+-------------+--------------+------------|
| _train_mnist_contrastive_a8d14_00000 | TERMINATED | 127.0.0.1:54772 |         0.25 |          1   |         256 |   0.00149231 | 0.00449065 |
| _train_mnist_contrastive_a8d14_00001 | TERMINATED | 127.0.0.1:55949 |         0.5  |          1   |         256 |   0.0228083  | 0.0325808  |
| _train_mnist_contrastive_a8d14_00002 | TERMINATED | 127.0.0.1:57140 |         0.25 |          1.5 |         256 |   0          | 0          |
| _train_mnist_contrastive_a8d14_00003 | TERMINATED | 127.0.0.1:58260 |         0.5  |          1.5 |         256 |   0.00397437 | 0.0122953  |


TODO: Include a KNN classifier in the test and validation 
TODO: Transformations for MNIST for contrastive - 30 degree rotation, shift 0.25 of the height and width, shear up to 45, zoom in range 0.5 to 1.5

`python3 train.py -p avg -a gelu -l 0.01 --batch-size 2048 --pretrain --embedding 256 --pos-margin 1.5 --neg-margin 0`

- Common confusion between 3 and 8, 4 and 9. How can we improve this? - dropout?


### Useful Links
- https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/
- https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#tune-pytorch-lightning-ref
