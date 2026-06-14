# selection_pipeline.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# own modules
from pluto.clustering.clusters import QuestionClusterer, QuestionClusterResult
from pluto.selection.config_selector import BaseSelector, SelectionResult


@dataclass(frozen=True)
class ClusterSelection:
    cluster_id: int
    questions: list[dict[str, Any]]
    selection: SelectionResult


@dataclass(frozen=True)
class PipelineResult:
    cluster_result: QuestionClusterResult
    cluster_selections: dict[int, ClusterSelection]


class ClusterSelectionPipeline:
    """
    Simple pipeline:
    1. Cluster the questions by similarity.
    2. Run the selector independently for each cluster.
    3. Return one optimal configuration per cluster.
    """

    def __init__(self, clusterer: QuestionClusterer, selector: BaseSelector):
        self.clusterer = clusterer
        self.selector = selector

    def run(
        self,
        items: list[dict[str, Any]],
        n_clusters: int | None = None,
        min_clusters: int = 2,
        max_clusters: int = 8,
        visualize: bool = False,
        save_path: str | None = None,
        **selector_kwargs,
    ) -> PipelineResult:
        if len(items) == 0:
            raise ValueError("items must be a non-empty list.")

        if not isinstance(items[0], dict) or "question" not in items[0]:
            raise ValueError(
                "Each item must be a dictionary containing at least the key 'question'."
            )

        questions = [item["question"] for item in items]

        cluster_result = self.clusterer.cluster_questions(
            questions=questions,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            visualize=visualize,
            save_path=save_path,
        )

        cluster_to_items = defaultdict(list)
        for item, label in zip(items, cluster_result.labels, strict=False):
            cluster_to_items[int(label)].append(item)

        cluster_selections = {
            cluster_id: ClusterSelection(
                cluster_id=cluster_id,
                questions=cluster_items,
                selection=self.selector.select(cluster_items, **selector_kwargs),
            )
            for cluster_id, cluster_items in cluster_to_items.items()
        }
        return PipelineResult(
            cluster_result=cluster_result,
            cluster_selections=cluster_selections,
        )
