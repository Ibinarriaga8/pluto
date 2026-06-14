# cluster_predictor.py
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

# own modules
from pluto.clustering.embeders import QuestionEmbedder
from pluto.rag.utils import device


class ClusterPredictor(ABC):
    """
    Base interface for assigning a new question to an existing cluster configuration.
    """
    def __init__(
        self,
        embedder: QuestionEmbedder,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        self.embedder = embedder
        self.train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
        self.train_labels = np.asarray(train_labels)

        if len(self.train_embeddings) == 0:
            raise ValueError("train_embeddings must be non-empty.")
        if len(self.train_embeddings) != len(self.train_labels):
            raise ValueError("train_embeddings and train_labels must have the same length.")

    @abstractmethod
    def predict_cluster(self, question: str) -> int:
        """
        Assigns a new question to one of the existing clusters.
        """
        pass


class WeightedKNNClusterPredictor(ClusterPredictor):
    """
    Assigns a new question to a cluster using weighted kNN over clustered training questions.
    """
    def __init__(
        self,
        embedder: QuestionEmbedder,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        k: int = 5,
    ):
        super().__init__(embedder, train_embeddings, train_labels)
        self.k = k

        if self.k < 1:
            raise ValueError("k must be >= 1.")

    def predict_cluster(self, question: str) -> int:
        """
        Predicts the cluster for a new question by finding the k nearest training questions and using a weighted vote.

        question -> embedding -> distances to train_embeddings -> weighted vote among nearest neighbors' cluster labels
        """
        query_embedding = self.embedder.encode([question])[0]

        query = torch.tensor(query_embedding, dtype=torch.float32).to(device)
        train = torch.tensor(self.train_embeddings, dtype=torch.float32).to(device)
        labels = torch.tensor(self.train_labels).to(device)

        query = F.normalize(query, dim=0)
        train = F.normalize(train, dim=1)

        similarities = train @ query

        k = min(self.k, len(similarities))
        topk_values, topk_indices = torch.topk(similarities, k=k)

        topk_labels = labels[topk_indices]

        unique_labels = torch.unique(topk_labels)
        scores = {}

        for label in unique_labels:
            mask = topk_labels == label
            scores[int(label)] = float(topk_values[mask].sum())

        return max(scores, key=scores.get)
