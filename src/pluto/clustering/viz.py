# viz.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ClusterVisualizer:
    """Visualization helpers for clustered question embeddings."""

    @staticmethod
    def _colours(n: int) -> list:
        cmap = matplotlib.colormaps["tab10"].resampled(n)
        return [cmap(i) for i in range(n)]

    @staticmethod
    def _scatter_reduced(
        ax,
        reduced: np.ndarray,
        labels: np.ndarray,
        colours: list,
    ) -> None:
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=[colours[lbl]], label=f"Cluster {lbl}",
                s=60, alpha=0.75, edgecolors="white", linewidths=0.4,
            )
            centroid = reduced[mask].mean(axis=0)
            ax.scatter(
                *centroid, marker="X", s=180, c=[colours[lbl]],
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            ax.annotate(
                f"C{lbl}", centroid, fontsize=10, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
            )

    def plot_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        save_path: str | None = None,
    ) -> None:
        """PCA 2-D scatter coloured by cluster with centroid markers."""
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embeddings)
        var = pca.explained_variance_ratio_ * 100
        colours = self._colours(len(np.unique(labels)))

        fig, ax = plt.subplots(figsize=(9, 7))
        self._scatter_reduced(ax, reduced, labels, colours)
        ax.set_title(
            f"PCA — clusters  (PC1 {var[0]:.1f} %, PC2 {var[1]:.1f} %)",
            fontsize=14,
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=10)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        save_path: str | None = None,
        perplexity: int | None = None,
    ) -> None:
        """t-SNE 2-D scatter coloured by cluster with centroid markers."""
        perp = perplexity or min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        reduced = tsne.fit_transform(embeddings)
        colours = self._colours(len(np.unique(labels)))

        fig, ax = plt.subplots(figsize=(9, 7))
        self._scatter_reduced(ax, reduced, labels, colours)
        ax.set_title("t-SNE — clusters", fontsize=14)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(fontsize=10)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
