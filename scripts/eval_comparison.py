#!/usr/bin/env python3
"""
eval_comparison.py — Three-way comparison of RAG configuration strategies.

Methods evaluated:
  1. MAB-clustered : cluster-conditioned routing via OnlineRAGPipeline
  2. Baseline      : fixed config (chunk_size=200, temperature=0.3)
  3. MAB-flat      : best single config derived from artifact (no routing);
                     proxy for MAB trained on all questions without clustering

Usage:
    python scripts/eval_comparison.py [--artifact artifacts/nyc_artifact.pkl]
                                      [--n-questions 20] [--trials 20] [--seed 0]
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np

from pluto.eval.datasets import SquadV2Loader
from pluto.pipelines.online_pipeline import OnlineRAGPipeline
from pluto.pipelines.training_pipeline import (
    BASELINE_CHUNK_SIZE,
    BASELINE_TEMPERATURE,
    ClusterConfig,
    register_pickle_compat,
)
from pluto.rag.rag_interface import ChromaRAGInterface
from pluto.rl.rl import DatasetRewardScorer

# ── helpers ──────────────────────────────────────────────────────────────────

def _build_rag(base_config, chunk_size: int, temperature: float, base_path: str) -> ChromaRAGInterface:
    cfg = copy.deepcopy(base_config)
    cfg.chunk_size = chunk_size
    cfg.chunk_overlap = chunk_size // 5
    cfg.llm_temperature = temperature
    cfg.base_path = base_path
    return ChromaRAGInterface(config=cfg)


def _flat_mab_config(cluster_configs: dict[int, ClusterConfig]) -> tuple[int, float, float]:
    """
    Derive the best single (chunk_size, temperature) config from the artifact as a
    proxy for 'MAB trained on all questions without clustering'.

    Strategy: weighted vote by n_questions × best_score for each cluster's best action.
    The config with the highest weighted score is selected.
    """
    best_cs, best_t, best_vote = None, None, -1.0
    for cfg in cluster_configs.values():
        vote = cfg.n_questions * cfg.best_score
        if vote > best_vote:
            best_vote = vote
            best_cs = cfg.chunk_size
            best_t = cfg.llm_temperature
    return best_cs, best_t, best_vote


def _print_cluster_table(cluster_configs: dict[int, ClusterConfig]) -> None:
    MIN_RELIABLE_QUESTIONS = 5  # below this the MAB can't explore 60 arms adequately

    col_w = [9, 11, 10, 12, 14, 8]
    header = (
        f"{'Cluster':>{col_w[0]}} "
        f"{'Questions':>{col_w[1]}} "
        f"{'ChunkSize':>{col_w[2]}} "
        f"{'Temperature':>{col_w[3]}} "
        f"{'OfflineScore':>{col_w[4]}} "
        f"{'Warning':>{col_w[5]}}"
    )
    sep = "-" * (sum(col_w) + len(col_w))
    print(sep)
    print(header)
    print(sep)
    warnings = []
    for cid in sorted(cluster_configs):
        c = cluster_configs[cid]
        warn = "⚠ few" if c.n_questions < MIN_RELIABLE_QUESTIONS else ""
        if warn:
            warnings.append(
                f"  Cluster {cid}: only {c.n_questions} question(s) — MAB cannot "
                f"reliably explore {60} arms. Consider retraining with more questions."
            )
        print(
            f"{cid:>{col_w[0]}} "
            f"{c.n_questions:>{col_w[1]}} "
            f"{c.chunk_size:>{col_w[2]}} "
            f"{c.llm_temperature:>{col_w[3]}.1f} "
            f"{c.best_score:>{col_w[4]}.4f} "
            f"{warn:>{col_w[5]}}"
        )
    print(sep)
    for w in warnings:
        print(w)
    if warnings:
        print()


def _print_results_table(
    names: list[str],
    all_rewards: list[list[float]],
) -> None:
    n_trials = len(all_rewards[0])
    # per-trial wins (1 winner per trial; ties split evenly)
    wins = [0.0] * len(names)
    for t in range(n_trials):
        scores = [r[t] for r in all_rewards]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if np.isclose(s, best)]
        for i in winners:
            wins[i] += 1.0 / len(winners)

    col_w = [20, 8, 8, 8, 8, 8]
    header = (
        f"{'Method':<{col_w[0]}} "
        f"{'Mean':>{col_w[1]}} "
        f"{'Std':>{col_w[2]}} "
        f"{'Min':>{col_w[3]}} "
        f"{'Max':>{col_w[4]}} "
        f"{'Wins%':>{col_w[5]}}"
    )
    sep = "-" * (sum(col_w) + len(col_w))
    print(sep)
    print(header)
    print(sep)
    for name, rewards, w in zip(names, all_rewards, wins, strict=False):
        arr = np.array(rewards)
        print(
            f"{name:<{col_w[0]}} "
            f"{arr.mean():>{col_w[1]}.4f} "
            f"{arr.std():>{col_w[2]}.4f} "
            f"{arr.min():>{col_w[3]}.4f} "
            f"{arr.max():>{col_w[4]}.4f} "
            f"{100 * w / n_trials:>{col_w[5]}.1f}%"
        )
    print(sep)


def _plot(names: list[str], all_rewards: list[list[float]], title: str, save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        markers = ["o", "s", "^"]
        plt.figure(figsize=(11, 6))
        for name, rewards, m in zip(names, all_rewards, markers, strict=False):
            plt.plot(rewards, label=name, marker=m)
        plt.title(f"Average Reward per Trial — {title}")
        plt.xlabel("Trial")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved → {save_path}")
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping plot.")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", default="artifacts/offline_selection_artifact.pkl")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="output/reward_comparison.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    register_pickle_compat()

    # ── load artifact ─────────────────────────────────────────────────────────
    pipeline = OnlineRAGPipeline(artifact_path=args.artifact)
    artifact = pipeline.artifact
    article_title = artifact.title
    base_config = artifact.base_config

    print(f"\nArticle : {article_title}")
    print(f"Trials  : {args.trials}  |  Questions/trial: {args.n_questions}\n")

    # ── cluster table ─────────────────────────────────────────────────────────
    print("=== Cluster Configurations (offline training) ===")
    _print_cluster_table(artifact.cluster_configs)

    # ── MAB-flat config ───────────────────────────────────────────────────────
    flat_cs, flat_t, flat_vote = _flat_mab_config(artifact.cluster_configs)
    print(
        f"MAB-flat proxy config: chunk_size={flat_cs}, temperature={flat_t}  "
        f"(weighted vote score={flat_vote:.4f})\n"
        f"  NOTE: proper MAB-flat requires retraining with --store-heatmaps.\n"
        f"  Current proxy = cluster config weighted by n_questions × offline_score.\n"
    )

    # ── build fixed RAG instances ─────────────────────────────────────────────
    baseline_rag = _build_rag(base_config, chunk_size=BASELINE_CHUNK_SIZE, temperature=BASELINE_TEMPERATURE,
                              base_path="./data/chroma/baseline")
    flat_rag = _build_rag(base_config, chunk_size=flat_cs, temperature=flat_t,
                          base_path="./data/chroma/flat")

    # ── evaluation ────────────────────────────────────────────────────────────
    rewarder = DatasetRewardScorer()
    loader = SquadV2Loader()
    _, all_questions, _ = loader.load_article_by_title(article_title)
    answerable = [q for q in all_questions if q["answers"]["text"]]

    rewards_mab: list[float] = []
    rewards_baseline: list[float] = []
    rewards_flat: list[float] = []

    for trial in range(args.trials):
        sample = random.sample(answerable, k=min(args.n_questions, len(answerable)))

        trial_scores: dict[str, list[float]] = {"mab": [], "baseline": [], "flat": []}

        for item in sample:
            q = item["question"]
            gt = item["answers"]["text"][0]

            mab_result = pipeline.answer(q)
            mab_r = rewarder(mab_result.answer, gt)

            baseline_ans = baseline_rag.ask(q)
            baseline_r = rewarder(baseline_ans, gt)

            flat_ans = flat_rag.ask(q)
            flat_r = rewarder(flat_ans, gt)

            trial_scores["mab"].append(mab_r)
            trial_scores["baseline"].append(baseline_r)
            trial_scores["flat"].append(flat_r)

            print(
                f"  Q: {q[:65]}\n"
                f"    MAB-clustered [cs={mab_result.chunk_size:4d}, t={mab_result.llm_temperature:.1f}]: "
                f"{mab_r:.4f}\n"
                f"    Baseline      [cs={BASELINE_CHUNK_SIZE:4d}, t={BASELINE_TEMPERATURE:.1f}]:                        "
                f"{baseline_r:.4f}\n"
                f"    MAB-flat      [cs={flat_cs:4d}, t={flat_t:.1f}]:                        "
                f"{flat_r:.4f}"
            )

        avg_mab = float(np.mean(trial_scores["mab"]))
        avg_base = float(np.mean(trial_scores["baseline"]))
        avg_flat = float(np.mean(trial_scores["flat"]))

        rewards_mab.append(avg_mab)
        rewards_baseline.append(avg_base)
        rewards_flat.append(avg_flat)

        print(
            f"\nTrial {trial + 1:02d}/{args.trials} | "
            f"MAB-clustered: {avg_mab:.4f} | "
            f"Baseline: {avg_base:.4f} | "
            f"MAB-flat: {avg_flat:.4f}\n"
        )

    # ── final report ──────────────────────────────────────────────────────────
    names = ["MAB-clustered", "Baseline (fixed)", "MAB-flat"]
    all_rewards = [rewards_mab, rewards_baseline, rewards_flat]

    print("\n=== Final Results ===")
    _print_results_table(names, all_rewards)
    print()

    _plot(names, all_rewards, article_title, args.plot)


if __name__ == "__main__":
    main()
