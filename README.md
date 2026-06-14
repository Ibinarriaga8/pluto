# pluto

> **RAG hyperparameter optimisation with Multi-Armed Bandits** — a Python library that uses UCB bandits and agglomerative question clustering to find the best Retrieval-Augmented Generation configuration for each question cluster, without expensive grid search.

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](./pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1 — RAG pipeline](#1--rag-pipeline)
  - [2 — MAB configuration selector](#2--mab-configuration-selector)
  - [3 — Cluster-aware selection pipeline](#3--cluster-aware-selection-pipeline)
- [Module Reference](#module-reference)
- [Scripts](#scripts)
- [Notebooks](#notebooks)
- [Development](#development)
- [Version History](#version-history)
- [Contributing](#contributing)

---

## Overview

`pluto` solves a practical problem: **which RAG configuration (chunk size, LLM temperature, …) gives the best answers for a given set of questions?**

Instead of running an expensive brute-force grid search, it uses:

1. **Multi-Armed Bandits (UCB)** to efficiently explore the configuration space and converge on the best `(chunk_size, llm_temperature)` pair.
2. **Agglomerative question clustering** to group semantically similar questions, so each cluster gets its own optimal configuration rather than a single global one.
3. **A KNN predictor** that maps new, unseen questions to the nearest cluster — and therefore to the right configuration — at inference time with a single call.

---

## Key Concepts

| Concept | Description |
|---|---|
| **RAGConfig** | Dataclass holding every knob that controls the RAG pipeline (LLM provider, chunk size, temperature, …) |
| **RagAction** | A single point in the configuration search space `(chunk_size, llm_temperature)` |
| **MABSelector** | UCB bandit that evaluates actions on a question set and returns the best `(action, score)` |
| **ClusterSelectionPipeline** | Clusters questions → runs `MABSelector` per cluster → returns one optimal config per cluster |
| **ClusterConfigPredictor** | Routes a new question to the right cluster and retrieves the pre-computed optimal config |
| **ExperimentRunner** | Convenience wrapper for running brute-force / MAB experiments and accumulating heatmaps |

---

## Repository Structure

```
pluto/
├── src/pluto/               # Installable Python package
│   ├── rag/                 # RAG pipeline
│   │   ├── config.py        #   RAGConfig, LLMProvider
│   │   ├── rag.py           #   ChromaRAG end-to-end pipeline
│   │   ├── rag_interface.py #   ChromaRAGInterface (low-level)
│   │   ├── llm_loader.py    #   LLM factory (Ollama / Groq)
│   │   └── utils.py         #   Prompt templates, helpers
│   ├── rl/                  # Action space and reward scorers
│   │   └── rl.py            #   RagAction, DatasetRewardScorer, HumanRewardScorer
│   ├── clustering/          # Question clustering
│   │   ├── clusters.py      #   AgglomerativeQuestionClusterer
│   │   ├── embeders.py      #   QuestionEmbedder
│   │   ├── cluster_predictor.py  # WeightedKNNClusterPredictor
│   │   └── viz.py           #   Cluster visualisation helpers
│   ├── eval/                # Evaluation
│   │   ├── evaluators.py    #   RAGBruteForceEvaluator, RAGMABEvaluator
│   │   ├── experiment.py    #   ExperimentRunner
│   │   ├── datasets.py      #   SquadV2Loader
│   │   └── viz.py           #   RewardHeatMap, plot_time_comparison
│   ├── selection/           # Configuration selection
│   │   ├── config_selector.py    # BaseSelector, MABSelector, SelectionResult
│   │   ├── config_predictor.py   # ClusterConfigPredictor, RoutedSelection
│   │   └── selection_pipeline.py # ClusterSelectionPipeline, PipelineResult
│   └── pipelines/           # End-to-end entry points
│       ├── training_pipeline.py  # pluto-train (offline MAB training)
│       └── online_pipeline.py    # pluto-ask  (inference from artifact)
├── scripts/                 # Runnable experiment scripts
├── notebooks/               # Jupyter notebooks
├── tests/                   # pytest test suite
├── artifacts/               # Saved pipeline artifacts (.pkl)
└── docs/                    # Documentation
```

---

## Installation

**Requirements:** Python 3.10+, a running [Ollama](https://ollama.com/) instance (or a Groq API key).

### Editable install (recommended for development)

```sh
pip install -e ".[dev]"
```

### Production install

```sh
pip install .
```

> All runtime and development dependencies are declared in [`pyproject.toml`](./pyproject.toml). There is no `requirements.txt`.

---

## Quick Start

### 1 — RAG pipeline

```python
from pluto.rag.config import RAGConfig
from pluto.rag.rag import ChromaRAG
from pluto.rag.utils import SHORT_ANSWER_PROMPT

config = RAGConfig(
    texts=["Your long document goes here..."],
    llm_model_name="llama3.1:8b",
    llm_provider="ollama",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    custom_prompt_template=SHORT_ANSWER_PROMPT,
    chunk_size=350,
    llm_temperature=0.2,
)

rag = ChromaRAG(config)
answer = rag.query("What is the capital of France?")
print(answer)
```

### 2 — MAB configuration selector

Find the best RAG configuration for a set of questions using the UCB bandit:

```python
from pluto.rag.config import RAGConfig
from pluto.rl.rl import DatasetRewardScorer, RagAction
from pluto.selection.config_selector import MABSelector

action_space = [
    RagAction(chunk_size=k, llm_temperature=t)
    for k in [100, 300, 500, 700, 1000]
    for t in [0.0, 0.3, 0.7, 1.0]
]

selector = MABSelector(
    config_template=config,          # RAGConfig from step 1
    reward_scorer=DatasetRewardScorer(similarity_threshold=0.5),
    action_space=action_space,
    alpha=1.0,
    seed=42,
)

result = selector.select(questions=items, trials=100)
print("Best config:", result.best_action)
print("Best score: ", result.best_score)
```

### 3 — Cluster-aware selection pipeline

Group questions by topic and find the optimal configuration per cluster:

```python
from pluto.clustering.embeders import QuestionEmbedder
from pluto.clustering.clusters import AgglomerativeQuestionClusterer
from pluto.clustering.cluster_predictor import WeightedKNNClusterPredictor
from pluto.selection.selection_pipeline import ClusterSelectionPipeline

embedder  = QuestionEmbedder("sentence-transformers/all-mpnet-base-v2")
clusterer = AgglomerativeQuestionClusterer(embedder=embedder)

pipeline = ClusterSelectionPipeline(clusterer=clusterer, selector=selector)
pipeline_result = pipeline.run(
    items=items,            # list of {"question": ..., "answers": ...} dicts
    min_clusters=2,
    max_clusters=5,
    trials=100,
)

# Predict optimal config for a new question
predictor = WeightedKNNClusterPredictor(
    embedder=embedder,
    train_embeddings=pipeline_result.cluster_result.embeddings,
    train_labels=pipeline_result.cluster_result.labels,
    k=3,
)

cluster_id = predictor.predict_cluster("What year was Python created?")
optimal = pipeline_result.cluster_selections[cluster_id].selection
print("Optimal action:", optimal.best_action)
```

---

## Module Reference

<details>
<summary><strong>pluto.rag</strong> — RAG pipeline</summary>

| Symbol | Description |
|---|---|
| `RAGConfig` | Dataclass with all RAG hyperparameters |
| `LLMProvider` | Enum: `OLLAMA` \| `GROQ` |
| `ChromaRAG` | End-to-end RAG pipeline backed by ChromaDB |
| `ChromaRAGInterface` | Low-level index / retrieval interface |
| `SHORT_ANSWER_PROMPT` | Ready-made prompt template for short factual answers |

</details>

<details>
<summary><strong>pluto.rl</strong> — Reward scoring</summary>

| Symbol | Description |
|---|---|
| `RagAction` | `(chunk_size, llm_temperature)` action dataclass — the MAB action space |
| `DatasetRewardScorer` | Cosine-similarity reward against gold answers (sentence-transformers) |
| `HumanRewardScorer` | Interactive human-in-the-loop reward scoring |

</details>

<details>
<summary><strong>pluto.clustering</strong> — Question clustering</summary>

| Symbol | Description |
|---|---|
| `QuestionEmbedder` | Wraps a sentence-transformer to embed questions |
| `AgglomerativeQuestionClusterer` | Agglomerative clustering with optional silhouette auto-selection |
| `QuestionClusterResult` | Holds `labels`, `embeddings`, `n_clusters` |
| `WeightedKNNClusterPredictor` | KNN predictor that maps new questions to existing clusters |

</details>

<details>
<summary><strong>pluto.eval</strong> — Evaluation</summary>

| Symbol | Description |
|---|---|
| `RAGBruteForceEvaluator` | Full grid-search over `(chunk_sizes × temperatures)` |
| `RAGMABEvaluator` | UCB MAB evaluator — returns `MABEvaluationResult` with heatmaps and snapshots |
| `MABEvaluationResult` | `final_heatmap`, `snapshot_heatmaps`, `snapshot_counts` |
| `ExperimentRunner` | High-level wrapper for multi-question brute-force / MAB experiments |
| `SquadV2Loader` | Loads random articles from the SQuAD v2 dataset |
| `RewardHeatMap` | Seaborn heatmap for reward visualisation |
| `plot_time_comparison` | Bar chart comparing brute-force vs MAB runtimes |

</details>

<details>
<summary><strong>pluto.selection</strong> — Configuration selection</summary>

| Symbol | Description |
|---|---|
| `BaseSelector` | Abstract base class for all selectors |
| `MABSelector` | UCB-backed selector — evaluates each question independently and averages heatmaps |
| `SelectionResult` | `best_action`, `best_score`, `heatmap`, `raw_results` |
| `ClusterSelectionPipeline` | Clusters → per-cluster MABSelector → `PipelineResult` |
| `PipelineResult` | `cluster_result` + `cluster_selections` dict |
| `ClusterConfigPredictor` | Routes a new question to the pre-computed optimal config |
| `RoutedSelection` | `question`, `cluster_id`, `best_action`, `selection_result` |

</details>

---

## Scripts

Pre-built runnable scripts are in `scripts/`. Run them from the repository root after installing the package.

| Script | Description |
|---|---|
| `scripts/single_question_analysis.py` | Brute-force + MAB on a single random question; saves heatmaps to `output/` |
| `scripts/overall_analysis.py` | Brute-force + MAB over all questions of a random article; saves aggregate heatmaps |
| `scripts/test_selector.py` | Demonstrates `MABSelector` on a small question set |
| `scripts/test_predictor.py` | Demonstrates the full pipeline + `ClusterConfigPredictor` |
| `scripts/clustering_demo.py` | Standalone clustering visualisation demo |

```sh
# Example
python scripts/single_question_analysis.py
```

---

## Notebooks

Interactive tutorials are in [`notebooks/`](./notebooks/):

| Notebook | Description |
|---|---|
| [`mab_tutorial.ipynb`](./notebooks/mab_tutorial.ipynb) | Step-by-step Multi-Armed Bandit tutorial |
| [`ucb_tutorial.ipynb`](./notebooks/ucb_tutorial.ipynb) | Upper Confidence Bound deep-dive |

---

## Development

```sh
# Install all dev dependencies
pip install -e ".[dev]"

# Lint & format
ruff check .
ruff format .

# Run tests
pytest tests/
```

---

## Version History

### [0.1.0] — 2026-06-14 · _Initial Release_

**Core infrastructure**
- `RAGConfig` dataclass — unified control surface for all RAG hyperparameters (LLM provider, model, chunk size, temperature, embedding model, prompt template)
- `ChromaRAG` end-to-end pipeline and `ChromaRAGInterface` low-level retrieval layer backed by ChromaDB
- LLM factory supporting Ollama and Groq providers

**Reward scoring**
- `RagAction` action space (chunk size × LLM temperature)
- UCB Multi-Armed Bandit selector (`MABSelector`) with configurable exploration coefficient
- `DatasetRewardScorer` (cosine similarity) and `HumanRewardScorer` reward functions

**Question clustering**
- `QuestionEmbedder` wrapping sentence-transformers
- `AgglomerativeQuestionClusterer` with silhouette-based automatic cluster count selection
- `WeightedKNNClusterPredictor` for routing unseen questions to the nearest cluster

**Selection pipeline**
- `ClusterSelectionPipeline` — clusters questions and runs a per-cluster MAB selector
- `ClusterConfigPredictor` — single-call inference: question in, optimal `RagAction` out

**Evaluation & tooling**
- Brute-force and MAB evaluators with reward heatmaps and step-wise snapshots
- `ExperimentRunner` for multi-question, multi-method comparisons
- `SquadV2Loader` dataset adapter
- Runnable scripts and Jupyter notebooks
- Full pytest suite and GitHub Actions CI

---

## Contributing

Want to contribute to **pluto**?  
Open an issue or a pull request — all contributions are welcome.

Licensed under [Apache 2.0](./LICENSE).
