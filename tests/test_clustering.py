# test_clustering.py
"""Tests for QuestionEmbedder, AgglomerativeQuestionClusterer, WeightedKNNClusterPredictor."""
from unittest.mock import MagicMock

import numpy as np
import pytest

EMBED_DIM = 16  # matches conftest


# ─────────────────────────────────────────────────────────────────────────────
# QuestionEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class TestQuestionEmbedder:
    def test_instantiation_smoke(self):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder("sentence-transformers/all-mpnet-base-v2")
        assert emb is not None

    def test_encode_returns_ndarray(self):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        result = emb.encode(["What is AI?"])
        assert isinstance(result, np.ndarray)

    def test_encode_shape(self):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        result = emb.encode(["Q1", "Q2", "Q3"])
        assert result.shape[0] == 3

    def test_encode_dtype_float64(self):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        result = emb.encode(["question"])
        assert result.dtype == np.float64

    def test_encode_empty_list_raises(self):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        with pytest.raises(ValueError):
            emb.encode([])

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_encode_various_lengths(self, n):
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        questions = [f"Question {i}?" for i in range(n)]
        result = emb.encode(questions)
        assert result.shape[0] == n


# ─────────────────────────────────────────────────────────────────────────────
# AgglomerativeQuestionClusterer
# ─────────────────────────────────────────────────────────────────────────────

class TestAgglomerativeQuestionClusterer:
    def _make_clusterer(self):
        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        return AgglomerativeQuestionClusterer(embedder=emb)

    def _make_embedder_returning(self, embeddings: np.ndarray):
        """Returns a real QuestionEmbedder whose encode() returns a fixed array."""
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        emb.encode = MagicMock(return_value=embeddings)
        return emb

    def test_instantiation_smoke(self):
        clusterer = self._make_clusterer()
        assert clusterer is not None

    def test_ward_requires_euclidean(self):
        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        from pluto.clustering.embeders import QuestionEmbedder
        with pytest.raises(ValueError):
            AgglomerativeQuestionClusterer(
                embedder=QuestionEmbedder(),
                metric="cosine",
                linkage="ward",
            )

    def test_cluster_too_few_questions_raises(self):
        clusterer = self._make_clusterer()
        with pytest.raises(ValueError):
            clusterer.cluster_questions(["Only one question"])

    def test_cluster_returns_result(self, synthetic_embeddings):
        from pluto.clustering.clusters import QuestionClusterResult
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, n_clusters=2)

        assert isinstance(result, QuestionClusterResult)

    def test_cluster_labels_length_matches_questions(self, synthetic_embeddings):
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, n_clusters=2)

        assert len(result.labels) == len(questions)

    def test_cluster_with_explicit_n(self, synthetic_embeddings):
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, n_clusters=2)

        assert result.n_clusters == 2

    def test_cluster_mapping_keys_match_labels(self, synthetic_embeddings):
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, n_clusters=2)

        assert set(result.cluster_to_questions.keys()) == set(result.labels.tolist())

    def test_auto_cluster_selection(self, synthetic_embeddings):
        """Without n_clusters, silhouette auto-selection should pick k=2 for 2 tight clusters."""
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, min_clusters=2, max_clusters=4)

        assert result.n_clusters >= 2

    def test_questions_df_has_expected_columns(self, synthetic_embeddings):
        embeddings, _ = synthetic_embeddings
        questions = [f"Q{i}" for i in range(len(embeddings))]

        from pluto.clustering.clusters import AgglomerativeQuestionClusterer
        emb = self._make_embedder_returning(embeddings)
        clusterer = AgglomerativeQuestionClusterer(embedder=emb)
        result = clusterer.cluster_questions(questions, n_clusters=2)

        assert "question" in result.questions_df.columns
        assert "cluster" in result.questions_df.columns


# ─────────────────────────────────────────────────────────────────────────────
# WeightedKNNClusterPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedKNNClusterPredictor:
    def _make_predictor(self, embeddings, labels, k=3):
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder
        emb = QuestionEmbedder()
        emb.encode = MagicMock(return_value=embeddings[:1])
        return WeightedKNNClusterPredictor(
            embedder=emb,
            train_embeddings=embeddings,
            train_labels=labels,
            k=k,
        )

    def test_instantiation(self, synthetic_embeddings):
        embeddings, labels = synthetic_embeddings
        predictor = self._make_predictor(embeddings, labels)
        assert predictor is not None

    def test_empty_embeddings_raises(self):
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder
        with pytest.raises(ValueError):
            WeightedKNNClusterPredictor(
                embedder=QuestionEmbedder(),
                train_embeddings=np.array([]),
                train_labels=np.array([]),
            )

    def test_mismatched_lengths_raises(self, synthetic_embeddings):
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder
        embeddings, labels = synthetic_embeddings
        with pytest.raises(ValueError):
            WeightedKNNClusterPredictor(
                embedder=QuestionEmbedder(),
                train_embeddings=embeddings,
                train_labels=labels[:-1],  # length mismatch
            )

    def test_k_zero_raises(self, synthetic_embeddings):
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder
        embeddings, labels = synthetic_embeddings
        with pytest.raises(ValueError):
            WeightedKNNClusterPredictor(
                embedder=QuestionEmbedder(),
                train_embeddings=embeddings,
                train_labels=labels,
                k=0,
            )

    def test_predict_returns_int(self, synthetic_embeddings):
        embeddings, labels = synthetic_embeddings
        predictor = self._make_predictor(embeddings, labels)
        # encode will return the first embedding (cluster 0)
        result = predictor.predict_cluster("Any question?")
        assert isinstance(result, int)

    def test_predict_returns_valid_cluster_label(self, synthetic_embeddings):
        embeddings, labels = synthetic_embeddings
        predictor = self._make_predictor(embeddings, labels)
        result = predictor.predict_cluster("Any question?")
        assert result in set(labels.tolist())

    def test_cluster_0_prediction(self, synthetic_embeddings):
        """A query close to cluster 0 should be predicted as cluster 0."""
        embeddings, labels = synthetic_embeddings
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder

        emb = QuestionEmbedder()
        # Use a cluster-0-like embedding as the query
        cluster0_emb = embeddings[0:1]
        emb.encode = MagicMock(return_value=cluster0_emb)

        predictor = WeightedKNNClusterPredictor(
            embedder=emb,
            train_embeddings=embeddings,
            train_labels=labels,
            k=3,
        )
        result = predictor.predict_cluster("question near cluster 0")
        assert result == 0

    def test_cluster_1_prediction(self, synthetic_embeddings):
        """A query close to cluster 1 should be predicted as cluster 1."""
        embeddings, labels = synthetic_embeddings
        from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
        from pluto.clustering.embeders import QuestionEmbedder

        emb = QuestionEmbedder()
        cluster1_emb = embeddings[-1:]  # last embedding is from cluster 1
        emb.encode = MagicMock(return_value=cluster1_emb)

        predictor = WeightedKNNClusterPredictor(
            embedder=emb,
            train_embeddings=embeddings,
            train_labels=labels,
            k=3,
        )
        result = predictor.predict_cluster("question near cluster 1")
        assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# ClusterVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterVisualizer:
    def test_plot_smoke_no_save(self, synthetic_embeddings):
        import matplotlib
        matplotlib.use("Agg")
        from pluto.clustering.viz import ClusterVisualizer
        embeddings, labels = synthetic_embeddings
        viz = ClusterVisualizer()
        viz.plot_clusters(embeddings=embeddings, labels=labels, save_path=None)

    def test_plot_saves_file(self, synthetic_embeddings, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from pluto.clustering.viz import ClusterVisualizer
        embeddings, labels = synthetic_embeddings
        save_path = str(tmp_path / "clusters.png")
        viz = ClusterVisualizer()
        viz.plot_clusters(embeddings=embeddings, labels=labels, save_path=save_path)
        import os
        assert os.path.exists(save_path)
