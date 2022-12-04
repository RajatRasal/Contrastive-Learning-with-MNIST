## Setup

1. Add the following to the local `.git/config`:
```
[user]
    name = Rajat Rasal
    email = yugiohrajat1@gmail.com
```
1. `conda env update -f environment.yml`
1. `conda activate contrastive_learning_mnist`
1. `pre-commit install`
1. `poetry install`

## Results
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

| activation   | pooling   |     lr |   batch_size |   val_loss |
|--------------|-----------|--------|--------------|------------|
| relu         | max       | 0.01   |         1024 |   0.558239 |
| gelu         | max       | 0.01   |         1024 |   0.540568 |
| relu         | max       | 0.01   |         2048 |   0.528964 |
| gelu         | max       | 0.01   |         2048 |   0.522437 |
| relu         | max       | 0.001  |         1024 |   0.516201 |
| gelu         | max       | 0.001  |         1024 |   0.513625 |
| relu         | max       | 0.001  |         2048 |   0.527985 |
| gelu         | max       | 0.001  |         2048 |   0.52203  |
| relu         | max       | 0.0001 |         1024 |   0.624486 |
| gelu         | max       | 0.0001 |         1024 |   0.62843  |
| relu         | max       | 0.0001 |         2048 |   0.72802  |
| gelu         | max       | 0.0001 |         2048 |   0.726901 |
| relu         | avg       | 0.01   |         1024 |   0.594014 |
| gelu         | avg       | 0.01   |         1024 |   0.496038 |
| relu         | avg       | 0.01   |         2048 |   0.870534 |
| gelu         | avg       | 0.01   |         2048 |   0.482411 |
| relu         | avg       | 0.001  |         1024 |   0.486609 |
| gelu         | avg       | 0.001  |         1024 |   0.492692 |
| relu         | avg       | 0.001  |         2048 |   0.497761 |
| gelu         | avg       | 0.001  |         2048 |   0.49835  |
| relu         | avg       | 0.0001 |         1024 |   0.591419 |
| gelu         | avg       | 0.0001 |         1024 |   0.588001 |
| relu         | avg       | 0.0001 |         2048 |   0.686366 |
| gelu         | avg       | 0.0001 |         2048 |   0.687941 |


|   neg_margin |   pos_margin |   embedding |   train_loss |   val_loss |
|--------------|--------------|-------------|--------------|------------|
|         0.25 |          1   |         256 |   0.00149231 | 0.00449065 |
|         0.5  |          1   |         256 |   0.0228083  | 0.0325808  |
|         0.25 |          1.5 |         256 |   0          | 0          |
|         0.5  |          1.5 |         256 |   0.00397437 | 0.0122953  |


TODO: Include a KNN classifier in the test and validation 
TODO: Transformations for MNIST for contrastive - 30 degree rotation, shift 0.25 of the height and width, shear up to 45, zoom in range 0.5 to 1.5

`python3 -m src.train -p avg -a gelu -l 0.01 --batch-size 2048 --pretrain --embedding 256 --pos-margin 1.5 --neg-margin 0 --preprocess`

- Common confusion between 4 and 9. How can we improve this? - dropout? - NO causes more confusion
- Longer training - the confusion between 4 and 8 still present
- Bigger embeddings dim - possibly... 512

## Useful Links
- https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/
- https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#tune-pytorch-lightning-ref
- https://stackoverflow.com/questions/60517190/are-poetry-lock-files-os-independent
