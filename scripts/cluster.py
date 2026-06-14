#!/usr/bin/env python3
"""Print cluster assignments and best configurations from a training artifact."""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from pluto.pipelines.training_pipeline import register_pickle_compat


def load_artifact(path: Path):
    register_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def group_questions_by_cluster(artifact) -> dict[int, list[str]]:
    clusters: dict[int, list[str]] = defaultdict(list)
    for question, label in zip(artifact.questions, artifact.train_labels, strict=False):
        clusters[int(label)].append(question["question"])
    return dict(clusters)


def print_cluster_summary(cluster_id: int, questions: list[str], config) -> None:
    sep = "=" * 60
    print(f"\n{sep}\nCluster {cluster_id}\n{sep}")
    print("\nQuestions:")
    for idx, q in enumerate(questions, start=1):
        print(f"  {idx:02d}. {q}")
    print("\nBest Configuration:")
    print(f"  Chunk Size   : {config.chunk_size}")
    print(f"  Temperature  : {config.llm_temperature}")
    print(f"  Best Score   : {config.best_score:.4f}")
    print(f"  # Questions  : {config.n_questions}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", default="artifacts/offline_selection_artifact.pkl",
                   help="Path to the training artifact pickle file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(Path(args.artifact))
    clusters = group_questions_by_cluster(artifact)
    for cluster_id in sorted(clusters):
        print_cluster_summary(cluster_id, clusters[cluster_id], artifact.cluster_configs[cluster_id])


if __name__ == "__main__":
    main()
