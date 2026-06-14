# evaluators.py
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# own modules
from pluto.rag.rag_interface import ChromaRAGInterface

logger = logging.getLogger(__name__)


class RAGEvaluator(ABC):
    """Abstract base class defining the contract for RAG evaluation strategies."""

    def __init__(self, config_template, reward_scorer):
        self.config_template = config_template
        self.reward_scorer = reward_scorer

    @abstractmethod
    def evaluate(self, items: list, **kwargs):
        pass


class RAGBruteForceEvaluator(RAGEvaluator):
    """Exhaustive grid search over the LLM action space."""

    def evaluate(self, items, temps, ks, iterations=5):
        matrix = np.zeros((len(temps), len(ks)))
        interface = ChromaRAGInterface(self.config_template)

        for i, t in enumerate(temps):
            for j, c in enumerate(ks):
                logger.info("Testing config: temp=%.2f, chunk_size=%d", t, c)
                interface.update_parameters(chunk_size=c, temperature=t)

                rewards = []
                for _ in range(iterations):
                    sample = random.choice(items)
                    response = interface.ask(sample["question"])
                    reward = self.reward_scorer(response, sample["answers"]["text"][0])
                    rewards.append(reward)

                matrix[i, j] = np.mean(rewards)

        return pd.DataFrame(matrix, index=temps, columns=ks)


@dataclass
class MABEvaluationResult:
    final_heatmap: pd.DataFrame
    snapshot_heatmaps: dict[int, pd.DataFrame] = field(default_factory=dict)
    snapshot_counts: dict[int, pd.DataFrame] | None = None


class RAGBandit:
    """
    Bandit environment for RAG hyper-parameter tuning.

    Each arm corresponds to one action in action_space.
    Pulling an arm = apply that config, run RAG, return mean reward.
    """

    def __init__(self, config_template, reward_scorer, action_space):
        self.config_template = config_template
        self.reward_scorer = reward_scorer
        self.action_space = action_space
        self.K = len(action_space)
        self.interface = ChromaRAGInterface(self.config_template)

    def pull(self, arm_idx: int, items: list, batch_size: int = 5) -> float:
        action = self.action_space[arm_idx]

        self.interface.update_parameters(
            chunk_size=action.chunk_size,
            temperature=action.llm_temperature,
        )

        rewards = []
        for _ in range(batch_size):
            sample = random.choice(items)
            question = sample["question"]
            gold = sample["answers"]["text"][0]

            response = self.interface.ask(question)
            rewards.append(float(self.reward_scorer(response, gold)))

        return float(np.mean(rewards))


class RAGMABEvaluator(RAGEvaluator):
    """
    Multi-Armed Bandit with UCB for online tuning of discrete RAG hyper-parameters.
    Each arm = one (chunk_size, temperature) configuration from action_space.
    """

    def __init__(
        self,
        config_template,
        reward_scorer,
        action_space,
        alpha: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(config_template, reward_scorer)
        if len(action_space) == 0:
            raise ValueError("action_space must be a non-empty list of discrete configurations.")

        self.action_space = action_space
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.K = len(action_space)
        self.t = 0
        self.counts = np.zeros(self.K, dtype=np.int64)
        self.values = np.zeros(self.K, dtype=np.float64)

    def _select_arm(self) -> int:
        """
        UCB selection:
            argmax_a Q(a) + alpha * sqrt(log(t) / N(a))

        with mandatory initialization pulls (each arm at least once).
        """
        for a in range(self.K):
            if self.counts[a] == 0:
                return a

        t = self.t + 1
        bonus = self.alpha * np.sqrt(np.log(t) / self.counts)
        ucb = self.values + bonus

        max_ucb = np.max(ucb)
        candidates = np.flatnonzero(np.isclose(ucb, max_ucb))
        return int(self.rng.choice(candidates))

    def _update_arm(self, arm_idx: int, reward: float) -> None:
        """Incremental mean update: Q <- Q + (r - Q) / N"""
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] += (reward - self.values[arm_idx]) / n

    def _build_heatmap(self, data: np.ndarray) -> pd.DataFrame:
        temps = sorted({a.llm_temperature for a in self.action_space})
        cs = sorted({a.chunk_size for a in self.action_space})

        temp_to_i = {t: i for i, t in enumerate(temps)}
        c_to_j = {c: j for j, c in enumerate(cs)}

        matrix = np.full((len(temps), len(cs)), np.nan, dtype=float)

        for arm_idx, a in enumerate(self.action_space):
            i = temp_to_i[a.llm_temperature]
            j = c_to_j[a.chunk_size]
            matrix[i, j] = float(data[arm_idx])

        heatmap_df = pd.DataFrame(matrix, index=temps, columns=cs)
        heatmap_df.index.name = "temperature"
        heatmap_df.columns.name = "chunk_size"
        return heatmap_df

    def _build_heatmap_from_values(self) -> pd.DataFrame:
        return self._build_heatmap(self.values)

    def _build_heatmap_from_counts(self) -> pd.DataFrame:
        return self._build_heatmap(self.counts.astype(float))

    def evaluate(
        self,
        items: list,
        trials: int = 100,
        batch_size: int = 1,
        log_every: int = 25,
        bandit_seed: int = 0,
        snapshot_every: int | None = None,
        store_count_snapshots: bool = False,
    ) -> MABEvaluationResult:
        if trials <= 0:
            raise ValueError("trials must be > 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if len(items) == 0:
            raise ValueError("items must be a non-empty list.")
        if snapshot_every is not None and snapshot_every <= 0:
            raise ValueError("snapshot_every must be > 0 or None.")

        self.t = 0
        self.counts = np.zeros(self.K, dtype=np.int64)
        self.values = np.zeros(self.K, dtype=np.float64)

        random.seed(bandit_seed)
        self.rng = np.random.default_rng(self.seed)

        bandit = RAGBandit(self.config_template, self.reward_scorer, self.action_space)

        snapshot_heatmaps = {}
        snapshot_counts = {}

        for step in range(trials):
            arm_idx = self._select_arm()
            action = self.action_space[arm_idx]

            reward = bandit.pull(arm_idx, items, batch_size=batch_size)

            self._update_arm(arm_idx, reward)
            self.t += 1

            current_step = step + 1

            if snapshot_every is not None and current_step % snapshot_every == 0:
                snapshot_heatmaps[current_step] = self._build_heatmap_from_values().copy()
                if store_count_snapshots:
                    snapshot_counts[current_step] = self._build_heatmap_from_counts().copy()

            if log_every and current_step % log_every == 0:
                best_arm = int(np.argmax(self.values))
                best_action = self.action_space[best_arm]
                logger.info(
                    "[UCB] step=%d/%d | last_reward=%.4f | "
                    "chosen=(c=%d, t=%.2f) | best_est=(c=%d, t=%.2f, mean=%.4f)",
                    current_step, trials, reward,
                    action.chunk_size, action.llm_temperature,
                    best_action.chunk_size, best_action.llm_temperature, self.values[best_arm],
                )

        final_heatmap = self._build_heatmap_from_values()

        if snapshot_every is not None and trials not in snapshot_heatmaps:
            snapshot_heatmaps[trials] = final_heatmap.copy()
            if store_count_snapshots:
                snapshot_counts[trials] = self._build_heatmap_from_counts().copy()

        return MABEvaluationResult(
            final_heatmap=final_heatmap,
            snapshot_heatmaps=snapshot_heatmaps,
            snapshot_counts=snapshot_counts,
        )
