# test_evaluators.py
"""Tests for RAGBruteForceEvaluator, RAGMABEvaluator, and related bandit logic."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# own modules
from pluto.rl.rl import RagAction

CHUNK_SIZES = [100, 300]
TEMPERATURES = [0.0, 0.5]

ACTION_SPACE = [
    RagAction(chunk_size=cs, llm_temperature=t)
    for cs in CHUNK_SIZES
    for t in TEMPERATURES
]


def _make_items(n=3):
    return [
        {"question": f"Q{i}", "answers": {"text": [f"A{i}"]}}
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RAGBruteForceEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGBruteForceEvaluator:
    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_evaluate_returns_dataframe(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBruteForceEvaluator
        iface = MagicMock()
        iface.ask.return_value = "answer"
        mock_iface_cls.return_value = iface

        scorer = MagicMock(return_value=0.7)
        evaluator = RAGBruteForceEvaluator(
            config_template=MagicMock(),
            reward_scorer=scorer,
        )
        result = evaluator.evaluate(
            _make_items(3),
            temps=TEMPERATURES,
            ks=CHUNK_SIZES,
            iterations=2,
        )
        assert isinstance(result, pd.DataFrame)

    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_evaluate_shape(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBruteForceEvaluator
        iface = MagicMock()
        iface.ask.return_value = "answer"
        mock_iface_cls.return_value = iface

        scorer = MagicMock(return_value=0.5)
        evaluator = RAGBruteForceEvaluator(MagicMock(), scorer)
        df = evaluator.evaluate(_make_items(2), temps=TEMPERATURES, ks=CHUNK_SIZES, iterations=1)

        assert df.shape == (len(TEMPERATURES), len(CHUNK_SIZES))

    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_evaluate_calls_scorer(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBruteForceEvaluator
        iface = MagicMock()
        iface.ask.return_value = "Paris"
        mock_iface_cls.return_value = iface

        scorer = MagicMock(return_value=0.9)
        evaluator = RAGBruteForceEvaluator(MagicMock(), scorer)
        evaluator.evaluate(_make_items(2), temps=[0.0], ks=[100], iterations=1)

        scorer.assert_called()

    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_evaluate_updates_interface_params(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBruteForceEvaluator
        iface = MagicMock()
        iface.ask.return_value = "x"
        mock_iface_cls.return_value = iface

        evaluator = RAGBruteForceEvaluator(MagicMock(), MagicMock(return_value=0.5))
        evaluator.evaluate(_make_items(2), temps=TEMPERATURES, ks=CHUNK_SIZES, iterations=1)

        iface.update_parameters.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# MABEvaluationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestMABEvaluationResult:
    def test_instantiation(self):
        from pluto.eval.evaluators import MABEvaluationResult
        df = pd.DataFrame([[0.5, 0.6]], columns=[100, 300], index=[0.0])
        result = MABEvaluationResult(final_heatmap=df)
        assert result.final_heatmap is df

    def test_default_snapshots_empty(self):
        from pluto.eval.evaluators import MABEvaluationResult
        df = pd.DataFrame([[0.5]])
        result = MABEvaluationResult(final_heatmap=df)
        assert result.snapshot_heatmaps == {}
        assert result.snapshot_counts is None


# ─────────────────────────────────────────────────────────────────────────────
# RAGMABEvaluator: internal bandit logic (no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGMABEvaluatorBanditLogic:
    def _make_evaluator(self, alpha=1.0):
        from pluto.eval.evaluators import RAGMABEvaluator
        return RAGMABEvaluator(
            config_template=MagicMock(),
            reward_scorer=MagicMock(return_value=0.5),
            action_space=ACTION_SPACE,
            alpha=alpha,
            seed=42,
        )

    def test_instantiation_smoke(self):
        ev = self._make_evaluator()
        assert ev.K == len(ACTION_SPACE)

    def test_empty_action_space_raises(self):
        from pluto.eval.evaluators import RAGMABEvaluator
        with pytest.raises(ValueError):
            RAGMABEvaluator(MagicMock(), MagicMock(), action_space=[], seed=0)

    def test_select_arm_initialization_phase(self):
        """Each arm must be pulled at least once before UCB kicks in."""
        ev = self._make_evaluator()
        selected = set()
        for _ in range(ev.K):
            arm = ev._select_arm()
            selected.add(arm)
            ev._update_arm(arm, 0.5)
            ev.t += 1
        assert selected == set(range(ev.K))

    def test_update_arm_increments_count(self):
        ev = self._make_evaluator()
        ev._update_arm(0, 0.8)
        assert ev.counts[0] == 1

    def test_update_arm_running_mean(self):
        ev = self._make_evaluator()
        ev._update_arm(0, 1.0)
        ev._update_arm(0, 0.0)
        assert abs(ev.values[0] - 0.5) < 1e-9

    def test_update_arm_incremental_mean(self):
        ev = self._make_evaluator()
        rewards = [0.2, 0.4, 0.6, 0.8]
        for r in rewards:
            ev._update_arm(0, r)
        assert abs(ev.values[0] - np.mean(rewards)) < 1e-9

    def test_build_heatmap_shape(self):
        ev = self._make_evaluator()
        # Force known values
        for i, action in enumerate(ACTION_SPACE):
            ev.counts[i] = 1
            ev.values[i] = float(i) * 0.1
        heatmap = ev._build_heatmap_from_values()
        assert heatmap.shape == (len(TEMPERATURES), len(CHUNK_SIZES))

    def test_build_heatmap_index_is_temperature(self):
        ev = self._make_evaluator()
        for i in range(ev.K):
            ev.counts[i] = 1
        heatmap = ev._build_heatmap_from_values()
        assert heatmap.index.name == "temperature"

    def test_build_heatmap_columns_is_chunk_size(self):
        ev = self._make_evaluator()
        for i in range(ev.K):
            ev.counts[i] = 1
        heatmap = ev._build_heatmap_from_values()
        assert heatmap.columns.name == "chunk_size"

    def test_build_count_heatmap_shape(self):
        ev = self._make_evaluator()
        for i in range(ev.K):
            ev.counts[i] = i + 1
        count_hm = ev._build_heatmap_from_counts()
        assert count_hm.shape == (len(TEMPERATURES), len(CHUNK_SIZES))

    def test_ucb_exploration_bonus_decreases_known_arms(self):
        """After many pulls on arm 0, arm 0 bonus should be smaller than arm 1 (0 pulls)."""
        ev = self._make_evaluator()
        # Simulate: pull arm 0 many times, arm 1 zero times
        for _ in range(20):
            ev._update_arm(0, 0.5)
            ev.t += 1
        # Don't pull arm 1 at all — it gets mandatory pull via initialization
        ev.counts[1] = 0  # force zero for the test
        bonus_0 = ev.alpha * np.sqrt(np.log(ev.t + 1) / ev.counts[0])
        # Arm 1 has 0 counts so it will be selected via initialization path
        arm = ev._select_arm()
        assert arm == 1  # mandatory initialization pull


class TestRAGMABEvaluatorEvaluate:
    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_returns_result(self, mock_bandit_cls):
        from pluto.eval.evaluators import MABEvaluationResult, RAGMABEvaluator
        bandit = MagicMock()
        bandit.pull.return_value = 0.6
        mock_bandit_cls.return_value = bandit

        ev = RAGMABEvaluator(
            config_template=MagicMock(),
            reward_scorer=MagicMock(return_value=0.6),
            action_space=ACTION_SPACE,
            seed=0,
        )
        result = ev.evaluate(_make_items(3), trials=len(ACTION_SPACE) + 2)
        assert isinstance(result, MABEvaluationResult)

    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_heatmap_shape(self, mock_bandit_cls):
        from pluto.eval.evaluators import RAGMABEvaluator
        bandit = MagicMock()
        bandit.pull.return_value = 0.5
        mock_bandit_cls.return_value = bandit

        ev = RAGMABEvaluator(MagicMock(), MagicMock(return_value=0.5), ACTION_SPACE, seed=0)
        result = ev.evaluate(_make_items(2), trials=len(ACTION_SPACE) + 1)
        assert result.final_heatmap.shape == (len(TEMPERATURES), len(CHUNK_SIZES))

    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_zero_trials_raises(self, mock_bandit_cls):
        from pluto.eval.evaluators import RAGMABEvaluator
        ev = RAGMABEvaluator(MagicMock(), MagicMock(), ACTION_SPACE, seed=0)
        with pytest.raises(ValueError):
            ev.evaluate(_make_items(2), trials=0)

    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_empty_items_raises(self, mock_bandit_cls):
        from pluto.eval.evaluators import RAGMABEvaluator
        ev = RAGMABEvaluator(MagicMock(), MagicMock(), ACTION_SPACE, seed=0)
        with pytest.raises(ValueError):
            ev.evaluate([], trials=5)

    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_with_snapshots(self, mock_bandit_cls):
        from pluto.eval.evaluators import RAGMABEvaluator
        bandit = MagicMock()
        bandit.pull.return_value = 0.4
        mock_bandit_cls.return_value = bandit

        ev = RAGMABEvaluator(MagicMock(), MagicMock(return_value=0.4), ACTION_SPACE, seed=0)
        n = len(ACTION_SPACE)
        result = ev.evaluate(_make_items(2), trials=n + 2, snapshot_every=n)
        assert len(result.snapshot_heatmaps) >= 1

    @patch("pluto.eval.evaluators.RAGBandit")
    def test_evaluate_with_count_snapshots(self, mock_bandit_cls):
        from pluto.eval.evaluators import RAGMABEvaluator
        bandit = MagicMock()
        bandit.pull.return_value = 0.4
        mock_bandit_cls.return_value = bandit

        ev = RAGMABEvaluator(MagicMock(), MagicMock(return_value=0.4), ACTION_SPACE, seed=0)
        n = len(ACTION_SPACE)
        result = ev.evaluate(
            _make_items(2),
            trials=n + 2,
            snapshot_every=n,
            store_count_snapshots=True,
        )
        assert result.snapshot_counts is not None
        assert len(result.snapshot_counts) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# RAGBandit
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGBandit:
    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_pull_returns_float(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBandit
        iface = MagicMock()
        iface.ask.return_value = "answer"
        mock_iface_cls.return_value = iface

        scorer = MagicMock(return_value=0.7)
        bandit = RAGBandit(MagicMock(), scorer, ACTION_SPACE)
        result = bandit.pull(0, _make_items(3), batch_size=2)
        assert isinstance(result, float)

    @patch("pluto.eval.evaluators.ChromaRAGInterface")
    def test_pull_calls_scorer(self, mock_iface_cls):
        from pluto.eval.evaluators import RAGBandit
        iface = MagicMock()
        iface.ask.return_value = "answer"
        mock_iface_cls.return_value = iface

        scorer = MagicMock(return_value=0.5)
        bandit = RAGBandit(MagicMock(), scorer, ACTION_SPACE)
        bandit.pull(0, _make_items(3), batch_size=3)
        assert scorer.call_count == 3
