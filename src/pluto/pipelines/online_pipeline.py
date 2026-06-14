# online_pipeline.py
from __future__ import annotations

import argparse
import copy
import pickle
from dataclasses import dataclass
from pathlib import Path

# own modules
from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
from pluto.clustering.embeders import QuestionEmbedder
from pluto.pipelines.training_pipeline import TrainingSelectionArtifact
from pluto.rag.rag_interface import ChromaRAGInterface


@dataclass
class OnlinePredictionResult:
    question: str
    answer: str
    cluster_id: int
    chunk_size: int
    llm_temperature: float
    cluster_score: float


class OnlineRAGPipeline:
    def __init__(self, artifact_path: str | Path, k: int = 3) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact = self._load_artifact(self.artifact_path)

        self.embedder = QuestionEmbedder(self.artifact.embedding_model_name)
        self.predictor = WeightedKNNClusterPredictor(
            embedder=self.embedder,
            train_embeddings=self.artifact.train_embeddings,
            train_labels=self.artifact.train_labels,
            k=k,
        )

        self._rag_cache: dict[tuple[int, float], ChromaRAGInterface] = {}

    @staticmethod
    def _load_artifact(path: Path) -> TrainingSelectionArtifact:
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        with path.open("rb") as f:
            artifact = pickle.load(f)

        required_attributes = [
            "embedding_model_name",
            "train_embeddings",
            "train_labels",
            "cluster_configs",
            "base_config",
        ]
        for attr in required_attributes:
            if not hasattr(artifact, attr):
                raise ValueError(f"Invalid artifact: missing attribute '{attr}'.")

        return artifact

    def _build_config_for_cluster(self, cluster_id: int):
        if cluster_id not in self.artifact.cluster_configs:
            available = sorted(self.artifact.cluster_configs.keys())
            raise KeyError(f"Cluster {cluster_id} not found. Available clusters: {available}")

        cluster_config = self.artifact.cluster_configs[cluster_id]
        rag_config = copy.deepcopy(self.artifact.base_config)

        rag_config.texts = [self.artifact.long_context]
        rag_config.chunk_size = int(cluster_config.chunk_size)
        rag_config.chunk_overlap = int(cluster_config.chunk_size) // 5  # match training formula in update_parameters
        rag_config.llm_temperature = float(cluster_config.llm_temperature)

        return rag_config, cluster_config

    def _get_rag(self, chunk_size: int, llm_temperature: float, rag_config) -> ChromaRAGInterface:
        cache_key = (int(chunk_size), float(llm_temperature))
        if cache_key not in self._rag_cache:
            self._rag_cache[cache_key] = ChromaRAGInterface(rag_config)
        return self._rag_cache[cache_key]

    def answer(self, question: str) -> OnlinePredictionResult:
        cluster_id = self.predictor.predict_cluster(question)
        rag_config, cluster_config = self._build_config_for_cluster(cluster_id)
        rag = self._get_rag(
            chunk_size=cluster_config.chunk_size,
            llm_temperature=cluster_config.llm_temperature,
            rag_config=rag_config,
        )

        answer = rag.ask(question)

        return OnlinePredictionResult(
            question=question,
            answer=answer,
            cluster_id=int(cluster_id),
            chunk_size=int(cluster_config.chunk_size),
            llm_temperature=float(cluster_config.llm_temperature),
            cluster_score=float(cluster_config.best_score),
        )


def run_online_stage(
    question: str,
    artifact_path: str = "artifacts/offline_selection_artifact.pkl",
    k: int = 3,
) -> OnlinePredictionResult:
    pipeline = OnlineRAGPipeline(
        artifact_path=artifact_path,
        k=k,
    )
    return pipeline.answer(question)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online stage for cluster-conditioned adaptive RAG.")
    parser.add_argument("--artifact-path", type=str, default="artifacts/offline_selection_artifact.pkl")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_online_stage(
        question=args.question,
        artifact_path=args.artifact_path,
        k=args.k,
    )

    print("\n--- Online RAG result ---")
    print(f"Question: {result.question}")
    print(f"Predicted cluster: {result.cluster_id}")
    print(
        "Selected configuration: "
        f"chunk_size={result.chunk_size}, "
        f"temperature={result.llm_temperature}, "
        f"offline_score={result.cluster_score:.4f}"
    )
    print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
