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

Pretraining
    Test Accuracy       │    0.9739999771118164   


### Useful Links
- https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry/
- https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#tune-pytorch-lightning-ref
