#!/usr/bin/env python3
"""Two-way evaluation: MAB-clustered pipeline vs. fixed-config baseline."""

import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt

from pluto.eval.datasets import SquadV2Loader
from pluto.pipelines.online_pipeline import OnlineRAGPipeline
from pluto.pipelines.training_pipeline import register_pickle_compat
from pluto.rag.rag_interface import ChromaRAGInterface
from pluto.rl.rl import DatasetRewardScorer

_BASELINE_CHUNK_SIZE = 400
_BASELINE_TEMPERATURE = 0.2
_BASELINE_DB = "./data/chroma/baseline"


def _build_baseline_rag(long_context: str, base_config) -> ChromaRAGInterface:
    config = copy.deepcopy(base_config)
    config.texts = [long_context]
    config.chunk_size = _BASELINE_CHUNK_SIZE
    config.chunk_overlap = _BASELINE_CHUNK_SIZE // 5
    config.llm_temperature = _BASELINE_TEMPERATURE
    config.base_path = _BASELINE_DB
    return ChromaRAGInterface(config=config)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", default="artifacts/offline_selection_artifact.pkl")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="output/reward_comparison.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    register_pickle_compat()

    pipeline = OnlineRAGPipeline(artifact_path=args.artifact)
    article_title = pipeline.artifact.title
    long_context = pipeline.artifact.long_context
    print(f"Evaluating on article: {article_title}")

    rewarder = DatasetRewardScorer()
    loader = SquadV2Loader()
    _, all_questions, _ = loader.load_article_by_title(article_title)
    answerable = [q for q in all_questions if q["answers"]["text"]]

    baseline_rag = _build_baseline_rag(long_context, pipeline.artifact.base_config)

    rewards_mab: list[float] = []
    rewards_baseline: list[float] = []

    for trial in range(args.trials):
        sample = random.sample(answerable, k=min(args.n_questions, len(answerable)))
        trial_mab = 0.0
        trial_base = 0.0

        for item in sample:
            q, gt = item["question"], item["answers"]["text"][0]
            result = pipeline.answer(q)
            mab_r = rewarder(result.answer, gt)
            base_r = rewarder(baseline_rag.ask(q), gt)
            trial_mab += mab_r
            trial_base += base_r
            print(
                f"  Q: {q[:60]}\n"
                f"    MAB  [cs={result.chunk_size}, t={result.llm_temperature}]: {mab_r:.4f}\n"
                f"    BASE [cs={_BASELINE_CHUNK_SIZE}, t={_BASELINE_TEMPERATURE}]: {base_r:.4f}"
            )

        n = len(sample)
        avg_mab = trial_mab / n
        avg_base = trial_base / n
        rewards_mab.append(avg_mab)
        rewards_baseline.append(avg_base)
        print(f"\nTrial {trial + 1:02d}/{args.trials} | MAB: {avg_mab:.4f} | Baseline: {avg_base:.4f}\n")

    print("\n=== Final Average Rewards ===")
    print(f"MAB Selector: {sum(rewards_mab) / len(rewards_mab):.4f}")
    print(f"Baseline:     {sum(rewards_baseline) / len(rewards_baseline):.4f}")

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_mab, label="MAB Selector", marker="o")
    plt.plot(rewards_baseline, label="Baseline RAG (fixed config)", marker="o")
    plt.title(f"Average Reward per Trial — {article_title}")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"Plot saved → {args.plot}")
    plt.show()


if __name__ == "__main__":
    main()
