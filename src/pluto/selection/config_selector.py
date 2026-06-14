# config_selector.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

# own modules
from pluto.eval.evaluators import MABEvaluationResult, RAGMABEvaluator


@dataclass(frozen=True)
class SelectionResult:
    best_action: Any
    best_score: float
    heatmap: pd.DataFrame
    raw_results: list[Any] | None = None


class BaseSelector(ABC):
    """
    Base contract for practical configuration selectors.
    """

    def __init__(self, config_template, reward_scorer, action_space: list[Any]):
        if len(action_space) == 0:
            raise ValueError("action_space must be a non-empty list.")

        self.config_template = config_template
        self.reward_scorer = reward_scorer
        self.action_space = action_space

    @abstractmethod
    def select(self, questions: list[dict[str, Any]], **kwargs) -> SelectionResult:
        raise NotImplementedError

    def _validate_questions(self, questions: list[Any]) -> None:
        if len(questions) == 0:
            raise ValueError("questions must be a non-empty list.")

        if not isinstance(questions[0], dict):
            raise ValueError(
                "questions must contain evaluable dataset items as dictionaries. "
                "Raw question strings are not enough to optimize a configuration "
                "because the selector needs reward feedback."
            )

        required_keys = {"question", "answers"}
        missing = required_keys.difference(questions[0].keys())
        if missing:
            raise ValueError(
                f"Each item must contain keys {sorted(required_keys)}; missing {sorted(missing)}."
            )

    def _best_action_from_heatmap(self, heatmap: pd.DataFrame) -> tuple[Any, float]:
        best_temp, best_chunk_size = heatmap.stack().idxmax()
        best_score = float(heatmap.loc[best_temp, best_chunk_size])

        for action in self.action_space:
            if (
                getattr(action, "llm_temperature", None) == best_temp
                and getattr(action, "chunk_size", None) == best_chunk_size
            ):
                return action, best_score

        raise ValueError("Best configuration found in heatmap is not present in action_space.")


class MABSelector(BaseSelector):
    """
    Selector backed by the UCB MAB evaluator.

    Runs a single bandit across all questions in the cluster, accumulating
    reward signal from every question on each arm pull.
    """

    def __init__(
        self,
        config_template,
        reward_scorer,
        action_space: list[Any],
        alpha: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(config_template, reward_scorer, action_space)
        self.alpha = float(alpha)
        self.seed = int(seed)

    def _build_evaluator(self) -> RAGMABEvaluator:
        return RAGMABEvaluator(
            config_template=self.config_template,
            reward_scorer=self.reward_scorer,
            action_space=self.action_space,
            alpha=self.alpha,
            seed=self.seed,
        )

    def select(
        self,
        questions: list[dict[str, Any]],
        trials: int = 100,
        batch_size: int = 1,
        log_every: int = 25,
        bandit_seed: int = 0,
        snapshot_every: int | None = None,
        store_count_snapshots: bool = False,
    ) -> SelectionResult:
        self._validate_questions(questions)

        evaluator = self._build_evaluator()
        result: MABEvaluationResult = evaluator.evaluate(
            items=questions,
            trials=trials,
            batch_size=batch_size,
            log_every=log_every,
            bandit_seed=bandit_seed,
            snapshot_every=snapshot_every,
            store_count_snapshots=store_count_snapshots,
        )

        best_action, best_score = self._best_action_from_heatmap(result.final_heatmap)

        return SelectionResult(
            best_action=best_action,
            best_score=best_score,
            heatmap=result.final_heatmap,
            raw_results=[result],
        )
