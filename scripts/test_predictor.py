import random

from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
from pluto.clustering.clusters import AgglomerativeQuestionClusterer
from pluto.clustering.embeders import QuestionEmbedder
from pluto.eval.datasets import SquadV2Loader
from pluto.rag.config import RAGConfig
from pluto.rag.utils import SHORT_ANSWER_PROMPT
from pluto.rl.rl import DatasetRewardScorer, RagAction
from pluto.selection.config_selector import MABSelector
from pluto.selection.selection_pipeline import ClusterSelectionPipeline

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

    questions = random.sample(items, k=min(10, len(items)))

    print(f"Article: {title}")
    print(f"Running pipeline on {len(questions)} questions...")

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

    embedder = QuestionEmbedder("sentence-transformers/all-mpnet-base-v2")
    clusterer = AgglomerativeQuestionClusterer(embedder=embedder)

    pipeline = ClusterSelectionPipeline(
        clusterer=clusterer,
        selector=selector,
    )

    pipeline_result = pipeline.run(
        items=questions,
        n_clusters=None,
        min_clusters=2,
        max_clusters=4,
        trials=100,
        batch_size=1,
        snapshot_every=25,
        store_count_snapshots=True,
    )

    print("\nCluster selections:")
    for cluster_id, cluster_selection in pipeline_result.cluster_selections.items():
        print(f"\nCluster {cluster_id}")
        print(f"Questions in cluster: {len(cluster_selection.questions)}")
        print("Best action:", cluster_selection.selection.best_action)
        print("Best score:", cluster_selection.selection.best_score)

    predictor = WeightedKNNClusterPredictor(
        embedder=embedder,
        train_embeddings=pipeline_result.cluster_result.embeddings,
        train_labels=pipeline_result.cluster_result.labels,
        k=3,
    )

    new_item = random.choice(items)
    new_question = new_item["question"]

    predicted_cluster = predictor.predict_cluster(new_question)
    optimal_selection = pipeline_result.cluster_selections[int(predicted_cluster)].selection

    print("\n--- Prediction for new question ---")
    print("Question:", new_question)
    print("Predicted cluster:", predicted_cluster)
    print("Optimal configuration:", optimal_selection.best_action)
    print("Configuration score:", optimal_selection.best_score)


if __name__ == "__main__":
    main()
