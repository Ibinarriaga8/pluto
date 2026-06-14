"""
Embedding study for offline_selection_artifact.pkl.

Produces:
  1. PCA 2-D scatter  — coloured by cluster, annotated with centroids
  2. t-SNE 2-D scatter — same colour scheme
  3. Inter/intra-cluster distance box-plot
  4. Per-cluster question table printed to stdout
  5. Silhouette score and per-sample silhouette bar chart
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

from pluto.clustering.viz import ClusterVisualizer

ARTIFACT_PATH = Path("artifacts/offline_selection_artifact.pkl")
OUTPUT_DIR = Path("output/embedding_study")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artifact(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_question_texts(questions: list) -> list[str]:
    return [
        q["question"] if isinstance(q, dict) else str(q)
        for q in questions
    ]


def _save(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


# ---------------------------------------------------------------------------
# Plot 3: intra / inter cluster distance distribution
# ---------------------------------------------------------------------------

def plot_distance_boxplot(embeddings: np.ndarray, labels: np.ndarray,
                          out_dir: Path) -> None:
    intra, inter = [], []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [intra, inter],
        tick_labels=["Intra-cluster", "Inter-cluster"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor("#4C9BE8")
    bp["boxes"][1].set_facecolor("#E87B4C")

    ax.set_title("Pairwise L2 distances — intra vs inter cluster", fontsize=13)
    ax.set_ylabel("Euclidean distance")
    fig.tight_layout()
    path = out_dir / "distance_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    ratio = np.mean(intra) / np.mean(inter)
    print(f"  Mean intra-cluster distance : {np.mean(intra):.4f}")
    print(f"  Mean inter-cluster distance : {np.mean(inter):.4f}")
    print(f"  Separation ratio (intra/inter): {ratio:.4f}  (lower = better separation)")


# ---------------------------------------------------------------------------
# Plot 4: per-sample silhouette bar chart
# ---------------------------------------------------------------------------

def plot_silhouette(embeddings: np.ndarray, labels: np.ndarray,
                    out_dir: Path) -> None:
    colours = ClusterVisualizer._colours(len(np.unique(labels)))
    if len(np.unique(labels)) < 2:
        print("  Silhouette: only one cluster, skipping.")
        return

    score = silhouette_score(embeddings, labels)
    sample_vals = silhouette_samples(embeddings, labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    y_lower = 10
    for lbl in np.unique(labels):
        vals = np.sort(sample_vals[labels == lbl])
        size = vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, vals,
            alpha=0.75, color=colours[lbl], label=f"Cluster {lbl}",
        )
        ax.text(-0.05, (y_lower + y_upper) / 2, f"C{lbl}", fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(score, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean silhouette = {score:.3f}")
    ax.set_title("Silhouette scores per sample", fontsize=13)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Questions (sorted within cluster)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "silhouette.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    print(f"  Global silhouette score: {score:.4f}")


# ---------------------------------------------------------------------------
# Stdout: per-cluster metrics table + question list
# ---------------------------------------------------------------------------

def _intra_distances(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = embeddings[mask]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    return np.array(dists) if dists else np.array([0.0])


def print_cluster_summary(
    question_texts: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    cluster_configs: dict,
) -> None:
    unique = np.unique(labels)
    n_total = len(question_texts)

    # Per-cluster silhouette (only meaningful with ≥2 clusters)
    if len(unique) >= 2:
        sil_samples = silhouette_samples(embeddings, labels)
    else:
        sil_samples = np.zeros(n_total)

    # Build rows
    rows = []
    for lbl in unique:
        cfg = cluster_configs.get(lbl)
        mask = labels == lbl
        n = int(mask.sum())
        pct = 100.0 * n / n_total
        sil_mean = sil_samples[mask].mean()
        sil_std  = sil_samples[mask].std()
        d = _intra_distances(embeddings, mask)
        rows.append({
            "Cluster":        lbl,
            "N":              n,
            "%":              pct,
            "Chunk size":     cfg.chunk_size,
            "Temperature":    cfg.llm_temperature,
            "Best score":     cfg.best_score,
            "Silhouette μ":   sil_mean,
            "Silhouette σ":   sil_std,
            "Intra-dist μ":   d.mean(),
            "Intra-dist σ":   d.std(),
        })

    # Column specs: (header, key, fmt, width)
    cols = [
        ("Cluster",      "Cluster",      "d",    7),
        ("N",            "N",            "d",    4),
        ("%",            "%",            ".1f",  6),
        ("Chunk",        "Chunk size",   "d",    6),
        ("Temp",         "Temperature",  ".1f",  5),
        ("Best score",   "Best score",   ".4f",  10),
        ("Sil μ",        "Silhouette μ", ".4f",  7),
        ("Sil σ",        "Silhouette σ", ".4f",  7),
        ("Intra μ",      "Intra-dist μ", ".4f",  8),
        ("Intra σ",      "Intra-dist σ", ".4f",  8),
    ]

    header = "  ".join(h.ljust(w) for h, _, _, w in cols)
    sep    = "  ".join("-" * w      for _, _, _, w in cols)

    print("\n" + "=" * len(sep))
    print(f"  CLUSTER METRICS  ({n_total} questions, {len(unique)} clusters)")
    print("=" * len(sep))
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for _, key, fmt, w in cols:
            val = row[key]
            cell = format(val, fmt)
            cells.append(cell.ljust(w))
        print("  ".join(cells))
    print("=" * len(sep))

    # Question list per cluster
    for lbl in unique:
        cfg  = cluster_configs.get(lbl)
        mask = labels == lbl
        qs   = [q for q, m in zip(question_texts, mask, strict=False) if m]
        print(f"\nCluster {lbl} — {int(mask.sum())} questions  "
              f"(chunk={cfg.chunk_size}, temp={cfg.llm_temperature})")
        print("-" * 60)
        for i, q in enumerate(qs, 1):
            print(f"  {i:2d}. {q}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading artifact from {ARTIFACT_PATH} …")
    art = load_artifact(ARTIFACT_PATH)

    embeddings = np.array(art.train_embeddings)   # (N, 768)
    labels = np.array(art.train_labels)           # (N,)
    question_texts = extract_question_texts(art.questions)
    n_clusters = art.n_clusters

    print(f"  Article  : {art.title}")
    print(f"  Questions: {len(question_texts)}")
    print(f"  Clusters : {n_clusters}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    out = OUTPUT_DIR
    viz = ClusterVisualizer()

    print("\n[1/4] PCA scatter …")
    pca_path = _save(out, "pca_clusters.png")
    viz.plot_clusters(embeddings, labels, save_path=str(pca_path))
    print(f"  Saved → {pca_path}")

    print("[2/4] t-SNE scatter …")
    tsne_path = _save(out, "tsne_clusters.png")
    viz.plot_tsne(embeddings, labels, save_path=str(tsne_path))
    print(f"  Saved → {tsne_path}")

    print("[3/4] Distance box-plot …")
    plot_distance_boxplot(embeddings, labels, out)

    print("[4/4] Silhouette chart …")
    plot_silhouette(embeddings, labels, out)

    print_cluster_summary(question_texts, labels, embeddings, art.cluster_configs)
    print(f"All figures saved under {out}/")


if __name__ == "__main__":
    main()
