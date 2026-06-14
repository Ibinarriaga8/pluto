# experiment.py
import shutil
import time

import pandas as pd

# own modules
from pluto.eval.evaluators import RAGBruteForceEvaluator, RAGMABEvaluator
from pluto.eval.viz import RewardHeatMap
from pluto.rag.config import LLMProvider, RAGConfig
from pluto.rag.utils import SHORT_ANSWER_PROMPT
from pluto.rl.rl import DatasetRewardScorer, RagAction


class ExperimentRunner:
    def __init__(
        self,
        verbose: bool,
        long_context: str,
        chunk_sizes: list[int],
        temperatures: list[float],
    ):
        self.chunk_sizes = chunk_sizes
        self.temperatures = temperatures

        shutil.rmtree("./eval_db", ignore_errors=True)
        shutil.rmtree("./rl_eval_db", ignore_errors=True)

        self.base_config = RAGConfig(
            texts=[long_context],
            verbose=verbose,
            llm_model_name="llama3.1:8b",
            llm_provider=LLMProvider.OLLAMA,
            embedding_model_name="sentence-transformers/all-mpnet-base-v2",
            custom_prompt_template=SHORT_ANSWER_PROMPT,
            persist_directory="./eval_db",
            llm_temperature=0.7,
            chunk_size=700,
            llm_top_k=10,
        )

        self.scorer = DatasetRewardScorer()

        self.action_space = [
            RagAction(chunk_size=k, llm_temperature=t)
            for k in self.chunk_sizes
            for t in self.temperatures
        ]

    def build_brute_force_evaluator(self):
        return RAGBruteForceEvaluator(
            config_template=self.base_config,
            reward_scorer=self.scorer,
        )

    def build_mab_evaluator(self):
        self.base_config.persist_directory = "./rl_eval_db"
        return RAGMABEvaluator(
            config_template=self.base_config,
            reward_scorer=self.scorer,
            action_space=self.action_space,
        )

    def run_brute_force(
        self,
        item,
        iterations: int,
        title: str,
        save_path: str,
    ):
        evaluator = self.build_brute_force_evaluator()

        start = time.time()
        heatmap_df = evaluator.evaluate(
            [item],
            ks=self.chunk_sizes,
            temps=self.temperatures,
            iterations=iterations,
        )
        elapsed = time.time() - start

        RewardHeatMap(title).plot(
            heatmap_df,
            save_path=save_path,
            elapsed=elapsed,
        )
        return heatmap_df, elapsed

    def run_mab(
        self,
        item,
        trials: int,
        title: str,
        save_path: str,
        snapshot_every: int | None = None,
        store_count_snapshots: bool = False,
    ):
        evaluator = self.build_mab_evaluator()

        start = time.time()
        result = evaluator.evaluate(
            [item],
            trials=trials,
            snapshot_every=snapshot_every,
            store_count_snapshots=store_count_snapshots,
        )
        elapsed = time.time() - start

        RewardHeatMap(title).plot(
            result.final_heatmap,
            save_path=save_path,
            elapsed=elapsed,
        )
        return result, elapsed

    def save_mab_snapshots(self, result, rewards_prefix: str, counts_prefix: str):
        for step, snapshot_heatmap_df in result.snapshot_heatmaps.items():
            RewardHeatMap(f"MAB Rewards - Step {step}").plot(
                snapshot_heatmap_df,
                save_path=f"{rewards_prefix}_{step}.png",
            )

        for step, count_heatmap_df in result.snapshot_counts.items():
            RewardHeatMap(f"MAB Counts - Step {step}").plot(
                count_heatmap_df,
                save_path=f"{counts_prefix}_{step}.png",
            )

    @staticmethod
    def accumulate_heatmap(total_df: pd.DataFrame | None, new_df: pd.DataFrame):
        if total_df is None:
            return new_df.copy()
        return total_df.add(new_df, fill_value=0)

    @staticmethod
    def compute_squared_error_heatmap(
        reference_df: pd.DataFrame,
        approx_df: pd.DataFrame,
    ):
        approx_df = approx_df.reindex_like(reference_df)

        return (approx_df - reference_df) ** 2
