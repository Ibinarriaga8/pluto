# pluto Documentation

See the main [README](../README.md) for a project overview.

## Structure

| Directory | Description |
|---|---|
| `src/pluto/` | Installable Python package |
| `scripts/` | Experiment and analysis scripts |
| `tests/` | pytest test suite |
| `artifacts/` | Saved training artifacts (`.pkl` files) |
| `output/` | Generated plots and evaluation outputs (gitignored) |
| `data/` | Runtime state: ChromaDB indexes (gitignored) |
| `notebooks/` | Jupyter notebooks |
| `docs/` | Documentation |

## Entry points

After `pip install -e .`:

```sh
mercury-train --help   # offline training pipeline
mercury-ask --question "..." --artifact-path artifacts/...pkl
```

## Modules

- **`pluto.rag`** – RAG pipeline (ChromaDB-backed and in-memory)
- **`pluto.rl`** – Action space and reward scorer
- **`pluto.clustering`** – Question embedder, clusterer, and KNN predictor
- **`pluto.eval`** – Evaluation strategies (brute-force / MAB) and visualisations
- **`pluto.selection`** – MAB selector and cluster selection pipeline
- **`pluto.pipelines`** – End-to-end offline training and online serving
