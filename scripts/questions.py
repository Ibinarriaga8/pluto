#!/usr/bin/env python3
"""Run the online RAG pipeline on the first N questions of a SQuAD v2 article."""

import argparse

from pluto.eval.datasets import SquadV2Loader
from pluto.pipelines.online_pipeline import run_online_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--title", default="Space_Race", help="SQuAD v2 article title")
    p.add_argument("--n", type=int, default=10, help="Number of questions to run")
    p.add_argument("--artifact", default="artifacts/offline_selection_artifact.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loader = SquadV2Loader()
    _, items, _ = loader.load_article_by_title(args.title)
    for item in items[:args.n]:
        result = run_online_stage(question=item["question"], artifact_path=args.artifact)
        print(
            f"\nQ: {item['question']}\n"
            f"A: {result.answer}\n"
            f"Cluster: {result.cluster_id} | cs={result.chunk_size}, "
            f"t={result.llm_temperature} | score={result.cluster_score:.4f}"
        )


if __name__ == "__main__":
    main()
