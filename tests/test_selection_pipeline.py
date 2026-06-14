# test_selection_pipeline.py
"""Tests for ClusterSelectionPipeline, ClusterConfigPredictor, and RoutedSelection."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# own modules
from pluto.rl.rl import RagAction
from pluto.selection.config_predictor import ClusterConfigPredictor, RoutedSelection
from pluto.selection.config_selector import SelectionResult
from pluto.selection.selection_pipeline import (
    ClusterSelection,
    ClusterSelectionPipeline,
    PipelineResult,
)

CHUNK_SIZES = [100, 300]
TEMPERATURES = [0.0, 0.5]

ACTION_SPACE = [
    RagAction(chunk_size=cs, llm_temperature=t)
    for cs in CHUNK_SIZES
    for t in TEMPERATURES
]


def _make_items(n=6):
    return [
        {"question": f"Q{i}", "answers": {"text": [f"A{i}"]}}
        for i in range(n)
    ]


def _make_heatmap():
    return pd.DataFrame(
        np.full((len(TEMPERATURES), len(CHUNK_SIZES)), 0.6),
        index=pd.Index(TEMPERATURES, name="temperature"),
        columns=pd.Index(CHUNK_SIZES, name="chunk_size"),
    )


def _make_selection_result(action=None):
    a = action or RagAction(chunk_size=300, llm_temperature=0.0)
    return SelectionResult(best_action=a, best_score=0.75, heatmap=_make_heatmap())


def _make_cluster_result(n_questions=6, n_clusters=2):
    """Builds a minimal QuestionClusterResult mock."""
    from pluto.clustering.clusters import QuestionClusterResult
    labels = np.array([i % n_clusters for i in range(n_questions)])
    questions = [f"Q{i}" for i in range(n_questions)]
    cluster_to_questions = {
        c: [q for q, l in zip(questions, labels) if l == c]
        for c in range(n_clusters)
    }
    return QuestionClusterResult(
        questions_df=MagicMock(),
        embeddings=np.random.rand(n_questions, 16),
        n_clusters=n_clusters,
        labels=labels,
        cluster_to_questions=cluster_to_questions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ClusterSelection
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterSelection:
    def test_instantiation(self):
        sel = ClusterSelection(
            cluster_id=0,
            questions=_make_items(3),
            selection=_make_selection_result(),
        )
        assert sel.cluster_id == 0

    def test_frozen(self):
        sel = ClusterSelection(0, _make_items(2), _make_selection_result())
        with pytest.raises((AttributeError, TypeError)):
            sel.cluster_id = 99  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_instantiation(self):
        cluster_result = _make_cluster_result()
        cluster_selections = {
            0: ClusterSelection(0, _make_items(3), _make_selection_result()),
            1: ClusterSelection(1, _make_items(3), _make_selection_result()),
        }
        result = PipelineResult(
            cluster_result=cluster_result,
            cluster_selections=cluster_selections,
        )
        assert result.cluster_result is cluster_result
        assert 0 in result.cluster_selections

    def test_frozen(self):
        cluster_result = _make_cluster_result()
        result = PipelineResult(cluster_result=cluster_result, cluster_selections={})
        with pytest.raises((AttributeError, TypeError)):
            result.cluster_selections = {}  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# ClusterSelectionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterSelectionPipeline:
    def _make_pipeline(self, n_clusters=2):
        cluster_result = _make_cluster_result(n_clusters=n_clusters)
        clusterer = MagicMock()
        clusterer.cluster_questions.return_value = cluster_result

        selector = MagicMock()
        selector.select.return_value = _make_selection_result()

        return ClusterSelectionPipeline(clusterer=clusterer, selector=selector), clusterer, selector

    def test_run_returns_pipeline_result(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run(_make_items(6))
        assert isinstance(result, PipelineResult)

    def test_run_cluster_result_stored(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run(_make_items(6))
        assert result.cluster_result is not None

    def test_run_creates_selection_per_cluster(self):
        pipeline, _, selector = self._make_pipeline(n_clusters=2)
        result = pipeline.run(_make_items(6))
        assert len(result.cluster_selections) == 2

    def test_run_selector_called_per_cluster(self):
        pipeline, _, selector = self._make_pipeline(n_clusters=3)
        items = _make_items(6)
        pipeline.run(items)
        assert selector.select.call_count == 3

    def test_run_clusterer_receives_questions(self):
        pipeline, clusterer, _ = self._make_pipeline()
        items = _make_items(6)
        pipeline.run(items)
        call_kwargs = clusterer.cluster_questions.call_args[1]
        assert len(call_kwargs["questions"]) == len(items)

    def test_run_empty_items_raises(self):
        pipeline, _, _ = self._make_pipeline()
        with pytest.raises(ValueError):
            pipeline.run([])

    def test_run_non_dict_items_raises(self):
        pipeline, _, _ = self._make_pipeline()
        with pytest.raises(ValueError):
            pipeline.run(["plain string"])

    def test_run_passes_selector_kwargs(self):
        pipeline, _, selector = self._make_pipeline()
        pipeline.run(_make_items(6), trials=50)
        _, kwargs = selector.select.call_args
        assert kwargs.get("trials") == 50

    def test_run_visualize_forwarded(self):
        pipeline, clusterer, _ = self._make_pipeline()
        pipeline.run(_make_items(6), visualize=False)
        call_kwargs = clusterer.cluster_questions.call_args[1]
        assert call_kwargs["visualize"] is False


# ─────────────────────────────────────────────────────────────────────────────
# RoutedSelection
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutedSelection:
    def test_instantiation(self):
        action = RagAction(300, 0.3)
        sel = RoutedSelection(
            question="Q?",
            cluster_id=1,
            best_action=action,
            selection_result=_make_selection_result(action),
        )
        assert sel.question == "Q?"
        assert sel.cluster_id == 1
        assert sel.best_action == action


# ─────────────────────────────────────────────────────────────────────────────
# ClusterConfigPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterConfigPredictor:
    def _make_predictor(self, n_clusters=2):
        action = RagAction(chunk_size=300, llm_temperature=0.0)
        cluster_selections = {
            c: ClusterSelection(c, _make_items(3), _make_selection_result(action))
            for c in range(n_clusters)
        }
        cluster_result = _make_cluster_result(n_clusters=n_clusters)
        pipeline_result = PipelineResult(
            cluster_result=cluster_result,
            cluster_selections=cluster_selections,
        )
        knn_predictor = MagicMock()
        knn_predictor.predict_cluster.return_value = 0
        return ClusterConfigPredictor(predictor=knn_predictor, pipeline_result=pipeline_result)

    def test_predict_cluster_returns_int(self):
        predictor = self._make_predictor()
        result = predictor.predict_cluster("Some question")
        assert isinstance(result, int)

    def test_predict_returns_routed_selection(self):
        predictor = self._make_predictor()
        result = predictor.predict("What is the capital of France?")
        assert isinstance(result, RoutedSelection)

    def test_predict_question_stored(self):
        predictor = self._make_predictor()
        result = predictor.predict("What is the capital of France?")
        assert result.question == "What is the capital of France?"

    def test_predict_cluster_id_matches(self):
        predictor = self._make_predictor()
        predictor.predictor.predict_cluster.return_value = 1
        result = predictor.predict("Some question")
        assert result.cluster_id == 1

    def test_predict_best_action_in_result(self):
        predictor = self._make_predictor()
        result = predictor.predict("question")
        assert isinstance(result.best_action, RagAction)

    def test_predict_unknown_cluster_raises(self):
        predictor = self._make_predictor(n_clusters=2)
        predictor.predictor.predict_cluster.return_value = 99
        with pytest.raises(ValueError, match="99"):
            predictor.predict("question for unknown cluster")

    def test_predict_cluster_delegates_to_inner(self):
        predictor = self._make_predictor()
        predictor.predict_cluster("question")
        predictor.predictor.predict_cluster.assert_called_with("question")
