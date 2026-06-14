# config_predictor.py
from dataclasses import dataclass
from typing import Any


@dataclass
class RoutedSelection:
    question: str
    cluster_id: int
    best_action: Any
    selection_result: Any


class ClusterConfigPredictor:
    def __init__(self, predictor, pipeline_result):
        self.predictor = predictor
        self.pipeline_result = pipeline_result

    def predict_cluster(self, question: str) -> int:
        return int(self.predictor.predict_cluster(question))

    def predict(self, question: str) -> RoutedSelection:
        cluster_id = self.predict_cluster(question)

        if cluster_id not in self.pipeline_result.cluster_selections:
            raise ValueError(f"Cluster {cluster_id} not found in pipeline result.")

        cluster_selection = self.pipeline_result.cluster_selections[cluster_id]

        return RoutedSelection(
            question=question,
            cluster_id=cluster_id,
            best_action=cluster_selection.selection.best_action,
            selection_result=cluster_selection.selection,
        )
