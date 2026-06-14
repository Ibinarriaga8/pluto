import os

from pluto.eval.datasets import SquadV2Loader
from pluto.eval.experiment import ExperimentRunner
from pluto.eval.viz import RewardHeatMap, plot_articles_time_comparison, plot_time_comparison

TEMPERATURES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
CHUNK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_random_article():
    print("Loading SQuAD v2 dataset...")
    loader = SquadV2Loader()
    title, items, long_context = loader.load_random_article()
    items = items[:10] # limit to 10 questions
    return title, items, long_context

def evaluate_article(title, items, long_context):

    print(f"Evaluating article: {title}")
    print(f"Number of questions: {len(items)}")

    experiment = ExperimentRunner(
        verbose=False,
        long_context=long_context,
        chunk_sizes=CHUNK_SIZES,
        temperatures=TEMPERATURES,
    )

    overall_brute_force_heatmap = None
    overall_mab_heatmap = None
    overall_mse_heatmap = None

    brute_force_times = []
    mab_times = []

    for i, item in enumerate(items, start=1):
        print(f"Processing question {i}/{len(items)}: {item['question']}")

        brute_force_heatmap_df, brute_force_time = experiment.run_brute_force(
            item=item,
            iterations=5,
            title=f"Brute Force Rewards - Q{i}",
            save_path=f"output/brute_force_rewards_heatmap_q{i}.png",
        )

        mab_result, mab_time = experiment.run_mab(
            item=item,
            trials=100,
            title=f"MAB Rewards - Q{i}",
            save_path=f"output/mab_rewards_heatmap_q{i}.png",
        )

        mab_heatmap_df = mab_result.final_heatmap

        brute_force_times.append(brute_force_time)
        mab_times.append(mab_time)

        overall_brute_force_heatmap = experiment.accumulate_heatmap(
            overall_brute_force_heatmap,
            brute_force_heatmap_df,
        )
        overall_mab_heatmap = experiment.accumulate_heatmap(
            overall_mab_heatmap,
            mab_heatmap_df,
        )

        mse_heatmap = experiment.compute_squared_error_heatmap(
            brute_force_heatmap_df,
            mab_heatmap_df,
        )
        overall_mse_heatmap = experiment.accumulate_heatmap(
            overall_mse_heatmap,
            mse_heatmap,
        )

    n_items = len(items)

    overall_brute_force_heatmap /= n_items
    overall_mab_heatmap /= n_items
    overall_mse_heatmap /= n_items

    RewardHeatMap("Overall Brute Force Rewards").plot(
        overall_brute_force_heatmap,
        save_path="output/overall_brute_force_rewards_heatmap.png",
    )

    RewardHeatMap("Overall MAB Rewards").plot(
        overall_mab_heatmap,
        save_path="output/overall_mab_rewards_heatmap.png",
    )

    RewardHeatMap("Overall Difference Heatmap").plot(
        overall_mab_heatmap - overall_brute_force_heatmap,
        save_path="output/overall_difference_heatmap.png",
        color_palette="coolwarm",
    )

    RewardHeatMap("Overall MSE Heatmap").plot(
        overall_mse_heatmap,
        save_path="output/overall_mse_heatmap.png",
        color_palette="coolwarm",
    )

    plot_time_comparison(
        brute_force_times,
        mab_times,
        save_path="output/time_comparison.png",
    )

    print("Mean squared error:", overall_mse_heatmap.values.mean())
    return brute_force_times, mab_times

def run_article():
    """
    runs the evaluation on a single article.
    """
    if os.exists("eval_db"):
        print("Cleaning up existing eval_db directory...")
        os.remove("eval_db")

    title, items, long_context = get_random_article()
    print(f"Running evaluation for {len(items)} questions in article '{title}'")
    brute_force_times, mab_times = evaluate_article(title, items, long_context)
    return brute_force_times, mab_times

def run_articles():
    """
    runs the evaluation on multiple articles and plots the average evaluation times for Brute Force and MAB across them.
    """
    bf_mean_times = []
    mab_mean_times = []

    for _ in range(20):
        brute_force_times, mab_times = run_article()
        bf_mean, mab_mean = sum(brute_force_times) / len(brute_force_times), sum(mab_times) / len(mab_times)

        bf_mean_times.append(bf_mean)
        mab_mean_times.append(mab_mean)


        plot_articles_time_comparison(
            bf_mean_times,
            mab_mean_times,
            save_path="output/articles_time_comparison.png",
        )







if __name__ == "__main__":
    run_article()
