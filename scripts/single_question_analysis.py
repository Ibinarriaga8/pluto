import random

from pluto.eval.datasets import SquadV2Loader
from pluto.eval.experiment import ExperimentRunner

TEMPERATURES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
CHUNK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def main():
    loader = SquadV2Loader()
    title, items, long_context = loader.load_random_article()
    print(f"Evaluating on Article: **{title}** with {len(items)} Q/A pairs.")
    item = random.choice(items)

    experiment = ExperimentRunner(
        long_context=long_context,
        chunk_sizes=CHUNK_SIZES,
        temperatures=TEMPERATURES,
    )

    experiment.run_brute_force(
        item=item,
        iterations=5,
        title="Brute Force Rewards",
        save_path="output/brute_force_rewards_heatmap.png",
    )

    result, _ = experiment.run_mab(
        item=item,
        trials=100,
        snapshot_every=25,
        store_count_snapshots=True,
        title="MAB Rewards",
        save_path="output/mab_rewards_heatmap.png",
    )

    experiment.save_mab_snapshots(
        result,
        rewards_prefix="output/mab_rewards_heatmap_step",
        counts_prefix="output/mab_counts_heatmap_step",
    )


if __name__ == "__main__":
    main()
