import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_proj(embeddings, labels, output_dir, seed=42):
    proj = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(
        x=proj[:, 0],
        y=proj[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10"),
    ).set(title="PCA")
    test_output_dir = os.path.join(output_dir, "test_diagrams/")
    os.makedirs(test_output_dir, exist_ok=True)
    plt.savefig(os.path.join(test_output_dir, "pca_proj.png"))
    plt.show()


def tsne_proj(embeddings, labels, output_dir, seed=42):
    proj = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(
        x=proj[:, 0],
        y=proj[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10"),
    ).set(title="T-SNE")
    test_output_dir = os.path.join(output_dir, "test_diagrams/")
    os.makedirs(test_output_dir, exist_ok=True)
    plt.savefig(os.path.join(test_output_dir, "tsne_proj.png"))
    plt.show()
