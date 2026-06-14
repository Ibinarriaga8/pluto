# training_pipeline.py
from __future__ import annotations

import argparse
import logging
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pluto.clustering.clusters import AgglomerativeQuestionClusterer
from pluto.clustering.embeders import QuestionEmbedder
from pluto.eval.datasets import SquadV2Loader
from pluto.rag.config import LLMProvider, RAGConfig
from pluto.rag.utils import SHORT_ANSWER_PROMPT
from pluto.rl.rl import DatasetRewardScorer, RagAction
from pluto.selection.config_selector import MABSelector
from pluto.selection.selection_pipeline import ClusterSelectionPipeline

logger = logging.getLogger(__name__)

TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
CHUNK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

BASELINE_CHUNK_SIZE = 200
BASELINE_TEMPERATURE = 0.3


@dataclass
class ClusterConfig:
    cluster_id: int
    chunk_size: int
    llm_temperature: float
    best_score: float
    n_questions: int


@dataclass
class TrainingSelectionArtifact:
    title: str
    long_context: str
    questions: list[dict[str, Any]]
    embedding_model_name: str
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    n_clusters: int
    cluster_configs: dict[int, ClusterConfig]
    base_config: RAGConfig


# When this module runs as __main__ (python -m ...), Python sets __module__
# on every class defined here to "__main__", so pickle can't find them on load.
# Pinning to the canonical module path fixes serialization permanently.
_MODULE = "pluto.pipelines.training_pipeline"
ClusterConfig.__module__ = _MODULE
TrainingSelectionArtifact.__module__ = _MODULE


def register_pickle_compat() -> None:
    """Register class aliases so old artifacts serialized as ``__main__.*`` can be loaded."""
    sys.modules["__main__"].TrainingSelectionArtifact = TrainingSelectionArtifact  # type: ignore[attr-defined]
    sys.modules["__main__"].ClusterConfig = ClusterConfig  # type: ignore[attr-defined]


def build_action_space() -> list[RagAction]:
    return [
        RagAction(chunk_size=chunk_size, llm_temperature=temperature)
        for chunk_size in CHUNK_SIZES
        for temperature in TEMPERATURES
    ]


def build_base_config(long_context: str, embedding_model_name: str) -> RAGConfig:
    return RAGConfig(
        texts=[long_context],
        llm_model_name="llama3.1:8b",
        llm_provider=LLMProvider.OLLAMA,
        embedding_model_name=embedding_model_name,
        custom_prompt_template=SHORT_ANSWER_PROMPT,
        base_path="./data/chroma/training",

        llm_temperature=0.2,
        llm_top_k=10,
    )


def build_cluster_configs(pipeline_result: Any) -> dict[int, ClusterConfig]:
    cluster_configs: dict[int, ClusterConfig] = {}

    for cluster_id, cluster_selection in pipeline_result.cluster_selections.items():

        best_action = cluster_selection.selection.best_action
        cluster_configs[int(cluster_id)] = ClusterConfig(
            cluster_id=int(cluster_id),
            chunk_size=int(best_action.chunk_size),
            llm_temperature=float(best_action.llm_temperature),
            best_score=float(cluster_selection.selection.best_score),
            n_questions=len(cluster_selection.questions),
        )

    return cluster_configs


def run_offline_stage(
    n_questions: int = 60,
    min_clusters: int = 2,
    max_clusters: int = 4,
    trials: int = 700,
    batch_size: int = 1,
    snapshot_every: int | None = 25,
    seed: int = 0,
    artifact_path: str = "artifacts/offline_selection_artifact.pkl",
    title: str | None = None,
) -> TrainingSelectionArtifact:

    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

    logger.info("Loading SQuAD v2 dataset...")
    loader = SquadV2Loader()

    if title is None:
        title, items, long_context = loader.load_random_article()
    else:
        title, items, long_context = loader.load_article_by_title(title)

    random.seed(seed)
    np.random.seed(seed)

    questions = random.sample(items, k=min(n_questions, len(items)))

    logger.info("Article: %s", title)
    logger.info("Offline optimization with %d questions.", len(questions))

    # Remove stale ChromaDB indexes from previous runs before building the config
    # so the training always starts from a clean state.
    shutil.rmtree("./data/chroma/training", ignore_errors=True)

    base_config = build_base_config(
        long_context=long_context,
        embedding_model_name=embedding_model_name,
    )

    scorer = DatasetRewardScorer()
    action_space = build_action_space()

    selector = MABSelector(
        config_template=base_config,
        reward_scorer=scorer,
        action_space=action_space,
        alpha=1.0,
        seed=seed,
    )

    embedder = QuestionEmbedder(embedding_model_name)
    clusterer = AgglomerativeQuestionClusterer(embedder=embedder)

    pipeline = ClusterSelectionPipeline(
        clusterer=clusterer,
        selector=selector,
    )

    pipeline_result = pipeline.run(
        items=questions,
        n_clusters=None,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        trials=trials,
        batch_size=batch_size,
        snapshot_every=snapshot_every,
        store_count_snapshots=True,
    )

    cluster_configs = build_cluster_configs(pipeline_result)

    clean_base_config = build_base_config(
        long_context=long_context,
        embedding_model_name=embedding_model_name,
    )

    artifact = TrainingSelectionArtifact(
        title=title,
        long_context=long_context,
        questions=questions,
        embedding_model_name=embedding_model_name,
        train_embeddings=pipeline_result.cluster_result.embeddings,
        train_labels=pipeline_result.cluster_result.labels,
        n_clusters=int(pipeline_result.cluster_result.n_clusters),
        cluster_configs=cluster_configs,
        base_config=clean_base_config,
    )

    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # When this module runs as __main__, its classes have __module__ set to the
    # canonical path but sys.modules only knows them as "__main__". Registering
    # the alias here lets pickle verify the class identity without re-importing
    # the module and getting a different class object.
    _canonical = "pluto.pipelines.training_pipeline"
    sys.modules.setdefault(_canonical, sys.modules[__name__])

    with output_path.open("wb") as f:
        pickle.dump(artifact, f)

    logger.info("Serialized training artifact saved at: %s", output_path)
    logger.info("Cluster configurations:")
    for cluster_id, config in sorted(cluster_configs.items()):
        logger.info(
            "Cluster %d | questions=%d | chunk_size=%d | temperature=%.2f | score=%.4f",
            cluster_id, config.n_questions, config.chunk_size, config.llm_temperature, config.best_score,
        )

    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training stage for cluster-conditioned adaptive RAG.")
    parser.add_argument("--title", type=str, default=None, help="Title of the article to load from SQuAD v2 (omit for random)")
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--trials", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-path", type=str, default="artifacts/offline_selection_artifact.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_offline_stage(
        n_questions=args.n_questions,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        trials=args.trials,
        batch_size=args.batch_size,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        artifact_path=args.artifact_path,
        title=args.title
    )


if __name__ == "__main__":
    main()
