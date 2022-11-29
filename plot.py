import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_proj(embeddings, labels, seed=42):
    proj = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(
        x=proj[:, 0],
        y=proj[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10"),
    ).set(title="PCA")
    plt.show()


def tsne_proj(embeddings, labels, seed=42):
    proj = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels).set(title="T-SNE")
    plt.show()
