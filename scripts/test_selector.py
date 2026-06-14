import random

from pluto.eval.datasets import SquadV2Loader
from pluto.rag.config import RAGConfig
from pluto.rag.utils import SHORT_ANSWER_PROMPT
from pluto.rl.rl import DatasetRewardScorer, RagAction
from pluto.selection.config_selector import MABSelector

TEMPERATURES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
CHUNK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def build_action_space() -> list[RagAction]:
    return [
        RagAction(chunk_size=k, llm_temperature=t)
        for k in CHUNK_SIZES
        for t in TEMPERATURES
    ]


def main() -> None:
    print("Loading SQuAD v2 dataset...")
    loader = SquadV2Loader()
    title, items, long_context = loader.load_random_article()

    questions = random.sample(items, k=min(5, len(items)))
    print(f"Article: {title}")
    print(f"Selecting configuration from {len(questions)} questions...")

    base_config = RAGConfig(
        texts=[long_context],
        llm_model_name="llama3.1:8b",
        llm_provider="ollama",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        custom_prompt_template=SHORT_ANSWER_PROMPT,
        persist_directory="./selector_eval_db",
        llm_temperature=0.2,
        llm_top_k=10,
    )

    scorer = DatasetRewardScorer()
    action_space = build_action_space()

    selector = MABSelector(
        config_template=base_config,
        reward_scorer=scorer,
        action_space=action_space,
        alpha=1.0,
        seed=0,
    )

    result = selector.select(
        questions=questions,
        trials=100,
        batch_size=1,
        snapshot_every=25,
        store_count_snapshots=True,
    )

    print("Best action from average heatmap:", result.best_action)
    print("Best score from average heatmap:", result.best_score)
    print("Average heatmap:")
    print(result.heatmap)

    if result.raw_results is not None:
        print("\nPer-question heatmaps:")
        for i, raw_result in enumerate(result.raw_results, start=1):
            print(f"\nQuestion {i}:", questions[i - 1]["question"])
            print(raw_result.final_heatmap)


if __name__ == "__main__":
    main()
