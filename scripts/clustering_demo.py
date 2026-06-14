from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
from pluto.clustering.clusters import AgglomerativeQuestionClusterer
from pluto.clustering.embeders import QuestionEmbedder
from pluto.clustering.viz import ClusterVisualizer
from pluto.eval.datasets import SquadV2Loader


def test_clustering():
    questions = [
        # Dinosaur Paleontology
        "What caused the extinction of the dinosaurs 65 million years ago?",
        "What did Tyrannosaurus rex eat during the Cretaceous period?",
        "How do paleontologists determine the age of fossils?",
        "What is the difference between herbivorous and carnivorous dinosaurs?",

        # Cryptocurrency / Blockchain
        "What is blockchain technology and how does it work?",
        "What is Bitcoin mining and why is it necessary?",
        "How do cryptocurrencies differ from traditional currencies?",
        "What is a smart contract in Ethereum?",

        # Classical Music
        "Who composed the Symphony No. 5 in C minor?",
        "What instruments are typically found in a symphony orchestra?",
        "What defines the Baroque period in classical music?",
        "How does a concerto differ from a symphony?",

        # Car Mechanics
        "How does an internal combustion engine work in a car?",
        "What causes a car battery to fail?",
        "What is the function of a transmission system in vehicles?",
        "How do disc brakes stop a vehicle?"
    ]

    embedder = QuestionEmbedder()
    visualizer = ClusterVisualizer()


    clusterer = AgglomerativeQuestionClusterer(
        embedder=embedder,
        visualizer=visualizer,
    )

    result = clusterer.cluster_questions(
        questions=questions,
        visualize=True,
        save_path="output/question_clusters.png",
    )
    predictor = WeightedKNNClusterPredictor(
    embedder=embedder,
    train_embeddings=result.embeddings,
    train_labels=result.labels,
    k=3,
    )

    question = "To me, a Tesla is better because it is an electric vehicle with advanced technology. What is the function of a transmission system in vehicles?"
    predicted_cluster = predictor.predict_cluster(question)

    print(result.questions_df)
    print(result.cluster_to_questions)
    print(f"Predicted cluster for question '{question}': {predicted_cluster}")


def main():
    # Load SQuAD v2 Dataset
    print("Loading SQuAD v2 dataset...")
    loader = SquadV2Loader()
    title, items, long_context = loader.load_random_article()

    questions = [item["question"] for item in items]

    questions = [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "What is the largest mammal?",
    ]
    embedder = QuestionEmbedder()
    visualizer = ClusterVisualizer()

    clusterer = AgglomerativeQuestionClusterer(
        embedder=embedder,
        visualizer=visualizer,
    )

    result = clusterer.cluster_questions(
        questions=questions,
        visualize=True,
        save_path="output/question_clusters.png",
    )

    print(result.questions_df)
    print(result.cluster_to_questions)


if __name__ == "__main__":
    test_clustering()
