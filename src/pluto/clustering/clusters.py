# clusters.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# own modules
from pluto.clustering.embeders import QuestionEmbedder
from pluto.clustering.viz import ClusterVisualizer


@dataclass
class QuestionClusterResult:
    """
    Stores the output of the clustering process.
    """
    questions_df: pd.DataFrame
    embeddings: np.ndarray
    n_clusters: int
    labels: np.ndarray
    cluster_to_questions: dict[int, list[str]] = field(default_factory=dict)


class QuestionClusterer(ABC):
    """
    Base interface for question clustering implementations.
    """
    def __init__(
        self,
        embedder: QuestionEmbedder,
        visualizer: ClusterVisualizer | None = None,
    ):
        self.embedder = embedder
        self.visualizer = visualizer

    def _build_cluster_mapping(
        self,
        questions: list[str],
        labels: np.ndarray,
    ) -> dict[int, list[str]]:
        cluster_to_questions: dict[int, list[str]] = {}
        for question, label in zip(questions, labels, strict=False):
            cluster_to_questions.setdefault(int(label), []).append(question)
        return cluster_to_questions

    @abstractmethod
    def cluster_questions(
        self,
        questions: list[str],
        n_clusters: int | None = None,
        min_clusters: int = 2,
        max_clusters: int = 8,
        visualize: bool = False,
        save_path: str | None = None,
    ) -> QuestionClusterResult:
        """
        Groups questions by similarity.
        """
        pass


class AgglomerativeQuestionClusterer(QuestionClusterer):
    """
    Groups questions by similarity using agglomerative clustering.
    """
    def __init__(
        self,
        embedder: QuestionEmbedder,
        visualizer: ClusterVisualizer | None = None,
        metric: str = "euclidean",
        linkage: str = "ward",
    ):
        super().__init__(embedder, visualizer)
        self.metric = metric
        self.linkage = linkage

        if self.linkage == "ward" and self.metric != "euclidean":
            raise ValueError("Ward linkage requires metric='euclidean'.")

    def _choose_n_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 8,
    ) -> int:
        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 questions to cluster.")
        if min_clusters < 2:
            raise ValueError("min_clusters must be >= 2.")
        if max_clusters < min_clusters:
            raise ValueError("max_clusters must be >= min_clusters.")

        best_k = 2
        best_score = -1.0

        upper_k = min(max_clusters, embeddings.shape[0] - 1)

        for k in range(min_clusters, upper_k + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric=self.metric,
                linkage=self.linkage,
            )
            labels = model.fit_predict(embeddings)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels, metric=self.metric)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def cluster_questions(
        self,
        questions: list[str],
        n_clusters: int | None = None,
        min_clusters: int = 2,
        max_clusters: int = 8,
        visualize: bool = False,
        save_path: str | None = None,
    ) -> QuestionClusterResult:
        """
        Automatically groups questions by similarity.

        If n_clusters is not provided, it is chosen automatically.
        """
        if len(questions) < 2:
            raise ValueError("Need at least 2 questions to cluster.")

        embeddings = self.embedder.encode(questions)

        if n_clusters is None:
            n_clusters = self._choose_n_clusters(
                embeddings=embeddings,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
            )

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=self.metric,
            linkage=self.linkage,
        )
        labels = model.fit_predict(embeddings)

        questions_df = pd.DataFrame({
            "question": questions,
            "cluster": labels
        })

        cluster_to_questions = self._build_cluster_mapping(questions, labels)

        result = QuestionClusterResult(
            questions_df=questions_df,
            embeddings=embeddings,
            n_clusters=n_clusters,
            labels=labels,
            cluster_to_questions=cluster_to_questions,
        )

        if visualize:
            if self.visualizer is None:
                raise ValueError("visualize=True requires a ClusterVisualizer instance.")
            self.visualizer.plot_clusters(
                embeddings=embeddings,
                labels=labels,
                save_path=save_path,
            )

        return result
