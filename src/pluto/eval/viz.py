# viz.py
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class RewardHeatMap:
    """Visualizes the reward matrix as a seaborn heatmap."""

    def __init__(self, title: str = "RAG Reward Heatmap: Temperature vs Chunk Size"):
        self.title = title

    def plot(
        self,
        results_df: pd.DataFrame,
        save_path: str = "output/reward_heatmap.png",
        elapsed: float | None = None,
        color_palette: str = "flare",
    ):
        """
        Generates and saves a heatmap from a results DataFrame.
        Expected format: Index = Temperature, Columns = Chunk Size, Values = Reward.
        """
        plt.figure(figsize=(10, 8))

        ax = sns.heatmap(
            results_df,
            annot=True,
            fmt=".3f",
            cmap=color_palette,
            linewidths=0.5,
            cbar_kws={"label": "Average Reward"},
        )

        plt.title(self.title, fontsize=15, pad=20)
        plt.xlabel("Chunk Size (Action Space)", fontsize=12)
        plt.ylabel("LLM Temperature (Action Space)", fontsize=12)
        if elapsed:
            plt.suptitle(f"Evaluation Time: {elapsed:.2f} seconds", fontsize=10, y=0.92, color="gray")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info("Heatmap saved to: %s", save_path)

        return ax


class TrajectoryDecorator(RewardHeatMap):
    """Overlays the agent's RL journey on a reward heatmap."""

    def __init__(self, component: RewardHeatMap, rl_history: dict, results_df: pd.DataFrame):
        self._component = component
        self.rl_history = rl_history
        self.results_df = results_df

    def plot(self, save_path: str = "rl_journey_heatmap.png"):
        ax = self._component.plot(self.results_df, save_path=None)

        temps = self.results_df.index.tolist()
        ks = self.results_df.columns.tolist()

        actions = list(self.rl_history.keys())

        x_coords = [ks.index(a.chunk_size) + 0.5 for a in actions]
        y_coords = [temps.index(a.llm_temperature) + 0.5 for a in actions]

        for i in range(len(x_coords) - 1):
            ax.annotate(
                "",
                xy=(x_coords[i + 1], y_coords[i + 1]),
                xytext=(x_coords[i], y_coords[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=2, alpha=0.8),
            )

        ax.scatter(x_coords[0], y_coords[0], color="blue", s=150, label="Start", zorder=5)
        ax.scatter(x_coords[-1], y_coords[-1], color="red", s=150, label="End", zorder=5)

        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()


def plot_time_comparison(brute_force_times, mab_times, save_path="output/time_comparison.png"):
    """Plots a line chart comparing Brute Force and MAB evaluation times across questions."""
    plt.figure(figsize=(10, 6))

    plt.plot(brute_force_times, label="Brute Force Time", color="blue")
    plt.plot(mab_times, label="MAB Time", color="orange")

    avg_brute_force_time = sum(brute_force_times) / len(brute_force_times)
    avg_mab_time = sum(mab_times) / len(mab_times)

    plt.axhline(avg_brute_force_time, color="blue", linestyle="--", label=f"Avg Brute Force Time: {avg_brute_force_time:.2f}s")
    plt.axhline(avg_mab_time, color="orange", linestyle="--", label=f"Avg MAB Time: {avg_mab_time:.2f}s")

    plt.title("Evaluation Time Comparison: Brute Force vs MAB", fontsize=14)
    plt.xlabel("Question Index", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.xticks(range(len(brute_force_times)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info("Time comparison chart saved to: %s", save_path)

    plt.show()
    plt.close()


def plot_articles_time_comparison(brute_force_times, mab_times, save_path="output/articles_time_comparison.png"):
    """Plots a line chart comparing average evaluation times across multiple articles."""
    plt.figure(figsize=(10, 6))

    plt.plot(brute_force_times, label="Brute Force Avg Time", color="blue")
    plt.plot(mab_times, label="MAB Avg Time", color="orange")

    avg_brute_force_time = sum(brute_force_times) / len(brute_force_times)
    avg_mab_time = sum(mab_times) / len(mab_times)

    plt.axhline(avg_brute_force_time, color="blue", linestyle="--", label=f"Overall Avg Brute Force Time: {avg_brute_force_time:.2f}s")
    plt.axhline(avg_mab_time, color="orange", linestyle="--", label=f"Overall Avg MAB Time: {avg_mab_time:.2f}s")

    plt.title("Average Evaluation Time Comparison Across Articles", fontsize=14)
    plt.xlabel("Article Index", fontsize=12)
    plt.ylabel("Average Time (seconds)", fontsize=12)
    plt.xticks(range(len(brute_force_times)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info("Articles time comparison chart saved to: %s", save_path)

    plt.show()
    plt.close()
