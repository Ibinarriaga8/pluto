# test_selection.py
"""Tests for BaseSelector, MABSelector, SelectionResult, and heatmap helpers."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# own modules
from pluto.rl.rl import RagAction
from pluto.selection.config_selector import SelectionResult

CHUNK_SIZES = [100, 300, 500]
TEMPERATURES = [0.0, 0.3, 0.7]

ACTION_SPACE = [
    RagAction(chunk_size=cs, llm_temperature=t)
    for cs in CHUNK_SIZES
    for t in TEMPERATURES
]


def _make_items(n=4):
    return [
        {"question": f"Q{i}", "answers": {"text": [f"A{i}"]}}
        for i in range(n)
    ]


def _make_heatmap(val=0.5):
    return pd.DataFrame(
        np.full((len(TEMPERATURES), len(CHUNK_SIZES)), val),
        index=pd.Index(TEMPERATURES, name="temperature"),
        columns=pd.Index(CHUNK_SIZES, name="chunk_size"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SelectionResult
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectionResult:
    def test_instantiation(self):
        action = RagAction(chunk_size=300, llm_temperature=0.3)
        hm = _make_heatmap()
        result = SelectionResult(best_action=action, best_score=0.8, heatmap=hm)
        assert result.best_action == action
        assert result.best_score == 0.8

    def test_frozen(self):
        result = SelectionResult(
            best_action=RagAction(100, 0.0),
            best_score=0.5,
            heatmap=_make_heatmap(),
        )
        with pytest.raises((AttributeError, TypeError)):
            result.best_score = 0.9  # type: ignore[misc]

    def test_raw_results_default_none(self):
        result = SelectionResult(RagAction(100, 0.0), 0.5, _make_heatmap())
        assert result.raw_results is None

    def test_raw_results_stored(self):
        raw = [MagicMock(), MagicMock()]
        result = SelectionResult(RagAction(100, 0.0), 0.5, _make_heatmap(), raw_results=raw)
        assert result.raw_results == raw


# ─────────────────────────────────────────────────────────────────────────────
# BaseSelector validation helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseSelectorValidation:
    def _make_selector(self):
        from pluto.selection.config_selector import MABSelector
        return MABSelector(
            config_template=MagicMock(),
            reward_scorer=MagicMock(),
            action_space=ACTION_SPACE,
        )

    def test_empty_action_space_raises(self):
        from pluto.selection.config_selector import MABSelector
        with pytest.raises(ValueError):
            MABSelector(MagicMock(), MagicMock(), action_space=[])

    def test_validate_empty_questions_raises(self):
        selector = self._make_selector()
        with pytest.raises(ValueError):
            selector._validate_questions([])

    def test_validate_non_dict_items_raises(self):
        selector = self._make_selector()
        with pytest.raises(ValueError):
            selector._validate_questions(["plain string"])

    def test_validate_missing_answers_key_raises(self):
        selector = self._make_selector()
        with pytest.raises(ValueError, match="answers"):
            selector._validate_questions([{"question": "Q?"}])

    def test_validate_missing_question_key_raises(self):
        selector = self._make_selector()
        with pytest.raises(ValueError, match="question"):
            selector._validate_questions([{"answers": {"text": ["A"]}}])

    def test_validate_valid_items_passes(self):
        selector = self._make_selector()
        selector._validate_questions(_make_items())  # should not raise


class TestBestActionFromHeatmap:
    def _selector(self):
        from pluto.selection.config_selector import MABSelector
        return MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)

    def test_best_action_is_in_action_space(self):
        selector = self._selector()
        hm = _make_heatmap(0.5)
        # Set one cell to the maximum
        hm.loc[0.3, 300] = 0.99
        action, score = selector._best_action_from_heatmap(hm)
        assert action in ACTION_SPACE

    def test_best_score_matches_heatmap_max(self):
        selector = self._selector()
        hm = _make_heatmap(0.5)
        hm.loc[0.7, 500] = 0.95
        _, score = selector._best_action_from_heatmap(hm)
        assert abs(score - 0.95) < 1e-9

    def test_best_action_chunk_and_temp(self):
        selector = self._selector()
        hm = _make_heatmap(0.5)
        hm.loc[0.0, 100] = 1.0
        action, _ = selector._best_action_from_heatmap(hm)
        assert action.chunk_size == 100
        assert action.llm_temperature == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MABSelector.select
# ─────────────────────────────────────────────────────────────────────────────

class TestMABSelectorSelect:
    def _make_mab_result(self, best_score=0.8):
        from pluto.eval.evaluators import MABEvaluationResult
        hm = _make_heatmap(0.5)
        hm.loc[0.3, 300] = best_score
        return MABEvaluationResult(final_heatmap=hm)

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_returns_selection_result(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = self._make_mab_result()
        mock_ev_cls.return_value = mock_ev

        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        result = selector.select(_make_items(2), trials=5)

        assert isinstance(result, SelectionResult)

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_best_action_in_action_space(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = self._make_mab_result()
        mock_ev_cls.return_value = mock_ev

        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        result = selector.select(_make_items(2), trials=5)

        assert result.best_action in ACTION_SPACE

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_best_score_is_float(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = self._make_mab_result()
        mock_ev_cls.return_value = mock_ev

        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        result = selector.select(_make_items(2), trials=5)

        assert isinstance(result.best_score, float)

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_heatmap_is_dataframe(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = self._make_mab_result()
        mock_ev_cls.return_value = mock_ev

        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        result = selector.select(_make_items(2), trials=5)

        assert isinstance(result.heatmap, pd.DataFrame)

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_builds_single_evaluator_for_cluster(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = self._make_mab_result()
        mock_ev_cls.return_value = mock_ev

        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        items = _make_items(4)
        selector.select(items, trials=5)

        # A single bandit runs across all questions in the cluster
        assert mock_ev_cls.call_count == 1
        # All items are passed to that single evaluator
        assert mock_ev.evaluate.call_args.kwargs["items"] == items

    @patch("pluto.selection.config_selector.RAGMABEvaluator")
    def test_select_empty_items_raises(self, mock_ev_cls):
        from pluto.selection.config_selector import MABSelector
        selector = MABSelector(MagicMock(), MagicMock(), ACTION_SPACE)
        with pytest.raises(ValueError):
            selector.select([], trials=5)
