# test_viz.py
"""Tests for RewardHeatMap, plot_time_comparison, ClusterVisualizer, TrajectoryDecorator."""
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for all viz tests


TEMPERATURES = [0.0, 0.3, 0.7]
CHUNK_SIZES = [100, 300, 500]


def _make_heatmap(val=0.5):
    return pd.DataFrame(
        np.full((len(TEMPERATURES), len(CHUNK_SIZES)), val),
        index=pd.Index(TEMPERATURES, name="temperature"),
        columns=pd.Index(CHUNK_SIZES, name="chunk_size"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RewardHeatMap
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardHeatMap:
    def test_instantiation_smoke(self):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap()
        assert hm is not None

    def test_custom_title(self):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap(title="My Custom Title")
        assert hm.title == "My Custom Title"

    def test_plot_no_save(self):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap()
        ax = hm.plot(_make_heatmap(), save_path=None)
        assert ax is not None

    def test_plot_saves_file(self, tmp_path):
        from pluto.eval.viz import RewardHeatMap
        save_path = str(tmp_path / "heatmap.png")
        hm = RewardHeatMap()
        hm.plot(_make_heatmap(), save_path=save_path)
        assert os.path.exists(save_path)

    def test_plot_with_time_annotation(self):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap()
        ax = hm.plot(_make_heatmap(), save_path=None, elapsed=12.5)
        assert ax is not None

    def test_plot_uniform_values(self):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap()
        ax = hm.plot(_make_heatmap(0.0), save_path=None)
        assert ax is not None

    @pytest.mark.parametrize("palette", ["flare", "viridis", "Blues"])
    def test_plot_different_palettes(self, palette):
        from pluto.eval.viz import RewardHeatMap
        hm = RewardHeatMap()
        ax = hm.plot(_make_heatmap(), save_path=None, color_palette=palette)
        assert ax is not None

    def test_plot_single_cell_heatmap(self):
        from pluto.eval.viz import RewardHeatMap
        single = pd.DataFrame([[0.7]], index=[0.0], columns=[100])
        hm = RewardHeatMap()
        ax = hm.plot(single, save_path=None)
        assert ax is not None


# ─────────────────────────────────────────────────────────────────────────────
# plot_time_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotTimeComparison:
    def test_smoke_no_save(self):
        from pluto.eval.viz import plot_time_comparison
        bf_times = [5.0, 6.0, 5.5]
        mab_times = [2.0, 1.8, 2.1]
        plot_time_comparison(bf_times, mab_times, save_path=None)

    def test_saves_file(self, tmp_path):
        from pluto.eval.viz import plot_time_comparison
        save_path = str(tmp_path / "times.png")
        plot_time_comparison([5.0, 6.0], [2.0, 1.5], save_path=save_path)
        assert os.path.exists(save_path)

    def test_single_data_point(self):
        from pluto.eval.viz import plot_time_comparison
        plot_time_comparison([5.0], [2.0], save_path=None)

    def test_many_data_points(self):
        from pluto.eval.viz import plot_time_comparison
        n = 20
        plot_time_comparison(
            [float(i) for i in range(n)],
            [float(i) * 0.5 for i in range(n)],
            save_path=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# plot_articles_time_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotArticlesTimeComparison:
    def test_smoke_no_save(self):
        from pluto.eval.viz import plot_articles_time_comparison
        plot_articles_time_comparison([4.0, 5.0, 3.0], [1.5, 2.0, 1.8], save_path=None)

    def test_saves_file(self, tmp_path):
        from pluto.eval.viz import plot_articles_time_comparison
        save_path = str(tmp_path / "articles_times.png")
        plot_articles_time_comparison([4.0, 5.0], [1.5, 2.0], save_path=save_path)
        assert os.path.exists(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# ClusterVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterVisualizerViz:
    def _make_data(self, n=12, dim=8):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(n, dim))
        labels = np.array([i % 3 for i in range(n)])
        return embeddings, labels

    def test_plot_no_save(self):
        from pluto.clustering.viz import ClusterVisualizer
        embeddings, labels = self._make_data()
        viz = ClusterVisualizer()
        viz.plot_clusters(embeddings=embeddings, labels=labels, save_path=None)

    def test_plot_saves_file(self, tmp_path):
        from pluto.clustering.viz import ClusterVisualizer
        embeddings, labels = self._make_data()
        save_path = str(tmp_path / "clusters.png")
        viz = ClusterVisualizer()
        viz.plot_clusters(embeddings=embeddings, labels=labels, save_path=save_path)
        assert os.path.exists(save_path)

    def test_plot_two_clusters(self):
        from pluto.clustering.viz import ClusterVisualizer
        embeddings, _ = self._make_data(10)
        labels = np.array([0] * 5 + [1] * 5)
        ClusterVisualizer().plot_clusters(embeddings, labels, save_path=None)

    def test_plot_many_dimensions_pca_reduces(self):
        from pluto.clustering.viz import ClusterVisualizer
        # PCA will reduce to 2 dims regardless of input dimensionality
        rng = np.random.default_rng(1)
        embeddings = rng.normal(size=(10, 768))
        labels = np.array([i % 2 for i in range(10)])
        ClusterVisualizer().plot_clusters(embeddings, labels, save_path=None)


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryDecorator
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryDecorator:
    def test_plot_smoke(self):
        from pluto.eval.viz import RewardHeatMap, TrajectoryDecorator
        from pluto.rl.rl import RagAction

        hm_df = _make_heatmap()
        base_hm = RewardHeatMap()

        # Build a fake rl_history: a dict mapping RagAction → reward
        actions = [
            RagAction(chunk_size=100, llm_temperature=0.0),
            RagAction(chunk_size=300, llm_temperature=0.3),
            RagAction(chunk_size=500, llm_temperature=0.7),
        ]
        rl_history = {a: float(i) * 0.3 for i, a in enumerate(actions)}

        decorator = TrajectoryDecorator(
            component=base_hm,
            rl_history=rl_history,
            results_df=hm_df,
        )
        decorator.plot(save_path=None)
