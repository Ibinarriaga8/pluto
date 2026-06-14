# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-06-14

### Added

**RAG pipeline**
- `RAGConfig` dataclass — unified control surface for LLM provider, model, chunk size, temperature, embedding model, and prompt template
- `ChromaRAG` end-to-end pipeline backed by ChromaDB
- `ChromaRAGInterface` low-level index / retrieval layer
- LLM factory (`llm_loader`) supporting Ollama and Groq providers
- Built-in prompt templates (`SHORT_ANSWER_PROMPT`)

**Multi-Armed Bandit selection**
- `RagAction` — action space defined by `(chunk_size, llm_temperature)`
- `MABSelector` — UCB bandit that explores the action space over a question set and returns the best configuration
- `DatasetRewardScorer` — cosine-similarity reward against gold answers via sentence-transformers
- `HumanRewardScorer` — interactive human-in-the-loop reward scoring

**Question clustering**
- `QuestionEmbedder` — wraps a sentence-transformer model to produce question embeddings
- `AgglomerativeQuestionClusterer` — agglomerative clustering with silhouette-based automatic cluster count selection
- `WeightedKNNClusterPredictor` — maps unseen questions to the nearest existing cluster

**Selection pipeline**
- `ClusterSelectionPipeline` — clusters a question set, runs a per-cluster `MABSelector`, and returns one optimal config per cluster
- `ClusterConfigPredictor` — single-call inference: question in, optimal `RagAction` out
- `SelectionResult`, `PipelineResult`, `RoutedSelection` result dataclasses

**Evaluation**
- `RAGBruteForceEvaluator` — exhaustive grid search over `(chunk_sizes × temperatures)`
- `RAGMABEvaluator` — UCB MAB evaluator with reward heatmaps and step-wise snapshots
- `ExperimentRunner` — high-level wrapper for multi-question brute-force / MAB comparisons
- `SquadV2Loader` — loads random articles and QA pairs from the SQuAD v2 dataset
- `RewardHeatMap`, `plot_time_comparison` — visualisation helpers

**Pipelines & entry points**
- `pluto-train` — offline training pipeline (clusters questions, runs MAB per cluster, saves artifact)
- `pluto-ask` — online inference (loads artifact, routes question, returns optimal config)

**Tooling**
- Full pytest suite (`tests/`)
- GitHub Actions CI — lint with Ruff + pytest on every push/PR to `main`
- `pyproject.toml` with `setuptools` build backend and `uv`-compatible dependency spec
