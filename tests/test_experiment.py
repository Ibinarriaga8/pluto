# test_experiment.py
"""Tests for ExperimentRunner (all external calls mocked)."""
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


CHUNK_SIZES = [100, 300, 500]
TEMPERATURES = [0.0, 0.3, 0.7]


def _make_heatmap(val=0.6):
    return pd.DataFrame(
        np.full((len(TEMPERATURES), len(CHUNK_SIZES)), val),
        index=pd.Index(TEMPERATURES, name="temperature"),
        columns=pd.Index(CHUNK_SIZES, name="chunk_size"),
    )


def _make_mab_result():
    from pluto.eval.evaluators import MABEvaluationResult
    return MABEvaluationResult(
        final_heatmap=_make_heatmap(),
        snapshot_heatmaps={50: _make_heatmap(0.4)},
        snapshot_counts={50: _make_heatmap(5.0)},
    )


def _make_runner(**kwargs):
    from pluto.eval.experiment import ExperimentRunner
    defaults = dict(
        verbose=False,
        long_context="Sample long text about France and its capital Paris.",
        chunk_sizes=CHUNK_SIZES,
        temperatures=TEMPERATURES,
    )
    defaults.update(kwargs)
    return ExperimentRunner(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner.__init__
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunnerInit:
    def test_instantiation_smoke(self):
        runner = _make_runner()
        assert runner is not None

    def test_chunk_sizes_stored(self):
        runner = _make_runner()
        assert runner.chunk_sizes == CHUNK_SIZES

    def test_temperatures_stored(self):
        runner = _make_runner()
        assert runner.temperatures == TEMPERATURES

    def test_action_space_size(self):
        runner = _make_runner()
        assert len(runner.action_space) == len(CHUNK_SIZES) * len(TEMPERATURES)

    def test_scorer_created(self):
        runner = _make_runner()
        assert runner.scorer is not None

    def test_base_config_created(self):
        runner = _make_runner()
        assert runner.base_config is not None

    def test_base_config_has_text(self):
        runner = _make_runner()
        assert len(runner.base_config.texts) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner.build_* methods
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunnerBuilders:
    def test_build_brute_force_evaluator(self):
        from pluto.eval.evaluators import RAGBruteForceEvaluator
        runner = _make_runner()
        ev = runner.build_brute_force_evaluator()
        assert isinstance(ev, RAGBruteForceEvaluator)

    def test_build_mab_evaluator(self):
        from pluto.eval.evaluators import RAGMABEvaluator
        runner = _make_runner()
        ev = runner.build_mab_evaluator()
        assert isinstance(ev, RAGMABEvaluator)

    def test_mab_evaluator_action_space_matches(self):
        runner = _make_runner()
        ev = runner.build_mab_evaluator()
        assert len(ev.action_space) == len(runner.action_space)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner.run_brute_force
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunnerBruteForce:
    @patch("pluto.eval.experiment.RewardHeatMap")
    @patch("pluto.eval.experiment.RAGBruteForceEvaluator")
    def test_run_brute_force_returns_tuple(self, mock_ev_cls, mock_hm_cls):
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = _make_heatmap()
        mock_ev_cls.return_value = mock_ev
        mock_hm_cls.return_value.plot.return_value = None

        runner = _make_runner()
        item = {"question": "Q?", "answers": {"text": ["A"]}}
        df, elapsed = runner.run_brute_force(item, iterations=1, title="T", save_path=None)

        assert isinstance(df, pd.DataFrame)
        assert elapsed >= 0

    @patch("pluto.eval.experiment.RewardHeatMap")
    @patch("pluto.eval.experiment.RAGBruteForceEvaluator")
    def test_run_brute_force_calls_evaluator(self, mock_ev_cls, mock_hm_cls):
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = _make_heatmap()
        mock_ev_cls.return_value = mock_ev
        mock_hm_cls.return_value.plot.return_value = None

        runner = _make_runner()
        item = {"question": "Q?", "answers": {"text": ["A"]}}
        runner.run_brute_force(item, iterations=2, title="T", save_path=None)

        mock_ev.evaluate.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner.run_mab
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunnerMAB:
    @patch("pluto.eval.experiment.RewardHeatMap")
    @patch("pluto.eval.experiment.RAGMABEvaluator")
    def test_run_mab_returns_tuple(self, mock_ev_cls, mock_hm_cls):
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = _make_mab_result()
        mock_ev_cls.return_value = mock_ev
        mock_hm_cls.return_value.plot.return_value = None

        runner = _make_runner()
        item = {"question": "Q?", "answers": {"text": ["A"]}}
        result, elapsed = runner.run_mab(item, trials=10, title="MAB", save_path=None)

        assert elapsed >= 0
        assert result is not None

    @patch("pluto.eval.experiment.RewardHeatMap")
    @patch("pluto.eval.experiment.RAGMABEvaluator")
    def test_run_mab_calls_evaluator(self, mock_ev_cls, mock_hm_cls):
        mock_ev = MagicMock()
        mock_ev.evaluate.return_value = _make_mab_result()
        mock_ev_cls.return_value = mock_ev
        mock_hm_cls.return_value.plot.return_value = None

        runner = _make_runner()
        item = {"question": "Q?", "answers": {"text": ["A"]}}
        runner.run_mab(item, trials=5, title="T", save_path=None)

        mock_ev.evaluate.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Static helper methods
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunnerStatics:
    def test_accumulate_heatmap_none_start(self):
        from pluto.eval.experiment import ExperimentRunner
        hm = _make_heatmap(0.5)
        result = ExperimentRunner.accumulate_heatmap(None, hm)
        np.testing.assert_array_equal(result.values, hm.values)

    def test_accumulate_heatmap_adds_values(self):
        from pluto.eval.experiment import ExperimentRunner
        hm1 = _make_heatmap(0.3)
        hm2 = _make_heatmap(0.5)
        result = ExperimentRunner.accumulate_heatmap(hm1, hm2)
        np.testing.assert_allclose(result.values, np.full_like(hm1.values, 0.8))

    def test_accumulate_does_not_mutate_original(self):
        from pluto.eval.experiment import ExperimentRunner
        hm1 = _make_heatmap(0.3)
        original_vals = hm1.values.copy()
        ExperimentRunner.accumulate_heatmap(hm1, _make_heatmap(0.5))
        np.testing.assert_array_equal(hm1.values, original_vals)

    def test_squared_error_heatmap_same_input(self):
        from pluto.eval.experiment import ExperimentRunner
        hm = _make_heatmap(0.5)
        error = ExperimentRunner.compute_squared_error_heatmap(hm, hm)
        np.testing.assert_allclose(error.values, 0.0)

    def test_squared_error_heatmap_constant_diff(self):
        from pluto.eval.experiment import ExperimentRunner
        ref = _make_heatmap(0.8)
        approx = _make_heatmap(0.6)
        error = ExperimentRunner.compute_squared_error_heatmap(ref, approx)
        np.testing.assert_allclose(error.values, np.full_like(ref.values, 0.04), atol=1e-9)

    def test_squared_error_shape_preserved(self):
        from pluto.eval.experiment import ExperimentRunner
        hm = _make_heatmap()
        error = ExperimentRunner.compute_squared_error_heatmap(hm, hm)
        assert error.shape == hm.shape

    @patch("pluto.eval.experiment.RewardHeatMap")
    def test_save_mab_snapshots_smoke(self, mock_hm_cls):
        mock_hm_cls.return_value.plot.return_value = None
        runner = _make_runner()
        result = _make_mab_result()
        runner.save_mab_snapshots(result, "rewards", "counts")
        assert mock_hm_cls.return_value.plot.call_count >= 2
