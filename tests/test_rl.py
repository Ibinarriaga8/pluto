# test_rl.py
"""Tests for RL components: RagAction, DatasetRewardScorer, HumanRewardScorer."""
from unittest.mock import patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# RagAction
# ─────────────────────────────────────────────────────────────────────────────

class TestRagAction:
    def test_creation(self):
        from pluto.rl.rl import RagAction
        a = RagAction(chunk_size=300, llm_temperature=0.5)
        assert a.chunk_size == 300
        assert a.llm_temperature == 0.5

    def test_frozen_immutable(self):
        from pluto.rl.rl import RagAction
        a = RagAction(chunk_size=300, llm_temperature=0.5)
        with pytest.raises((AttributeError, TypeError)):
            a.chunk_size = 999  # type: ignore[misc]

    def test_equality(self):
        from pluto.rl.rl import RagAction
        assert RagAction(100, 0.0) == RagAction(100, 0.0)

    def test_inequality(self):
        from pluto.rl.rl import RagAction
        assert RagAction(100, 0.0) != RagAction(200, 0.0)

    def test_hashable(self):
        from pluto.rl.rl import RagAction
        a = RagAction(300, 0.3)
        d = {a: "value"}
        assert d[a] == "value"

    @pytest.mark.parametrize("cs,t", [(50, 0.0), (100, 0.5), (500, 1.0)])
    def test_parametrized_creation(self, cs, t):
        from pluto.rl.rl import RagAction
        a = RagAction(chunk_size=cs, llm_temperature=t)
        assert a.chunk_size == cs
        assert a.llm_temperature == t


# ─────────────────────────────────────────────────────────────────────────────
# DatasetRewardScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetRewardScorer:
    def test_instantiation_smoke(self):
        from pluto.rl.rl import DatasetRewardScorer
        scorer = DatasetRewardScorer()
        assert scorer is not None

    def test_call_returns_float(self):
        from pluto.rl.rl import DatasetRewardScorer
        scorer = DatasetRewardScorer()
        result = scorer("Paris", "Paris")
        assert isinstance(result, float)

    def test_call_value_in_valid_range(self):
        from pluto.rl.rl import DatasetRewardScorer
        scorer = DatasetRewardScorer()
        result = scorer("Some answer", "Some gold")
        assert -1.0 <= result <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# HumanRewardScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanRewardScorer:
    def test_call_returns_valid_score(self):
        from pluto.rl.rl import HumanRewardScorer
        scorer = HumanRewardScorer()
        with patch("builtins.input", return_value="0.8"):
            result = scorer("answer", "gold")
        assert result == 0.8

    def test_retries_on_invalid_input(self):
        from pluto.rl.rl import HumanRewardScorer
        scorer = HumanRewardScorer()
        with patch("builtins.input", side_effect=["abc", "1.5", "0.6"]):
            result = scorer("answer", "gold")
        assert result == 0.6

    def test_boundary_value_zero(self):
        from pluto.rl.rl import HumanRewardScorer
        scorer = HumanRewardScorer()
        with patch("builtins.input", return_value="0.0"):
            result = scorer("a", "b")
        assert result == 0.0

    def test_boundary_value_one(self):
        from pluto.rl.rl import HumanRewardScorer
        scorer = HumanRewardScorer()
        with patch("builtins.input", return_value="1.0"):
            result = scorer("a", "b")
        assert result == 1.0
