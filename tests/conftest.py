# conftest.py
"""
Shared fixtures and global stubs for the pluto test suite.

Module-level sys.modules injections happen here, before any project import
can trigger expensive side-effects (model downloads, LLM server calls, etc.).
"""
import sys
from unittest.mock import MagicMock

import numpy as np
from langchain_core.messages import AIMessage
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM = 16  # tiny embedding size; keeps tests fast

SAMPLE_QUESTIONS = [
    {"question": "What is the capital of France?", "answers": {"text": ["Paris"]}},
    {"question": "When was Python created?", "answers": {"text": ["1991"]}},
    {"question": "Who wrote Hamlet?", "answers": {"text": ["Shakespeare"]}},
    {"question": "What is 2+2?", "answers": {"text": ["4"]}},
    {"question": "What planet is closest to the Sun?", "answers": {"text": ["Mercury"]}},
    {"question": "What is H2O?", "answers": {"text": ["water"]}},
]

CHUNK_SIZES = [100, 300, 500]
TEMPERATURES = [0.0, 0.3, 0.7]

# ─────────────────────────────────────────────────────────────────────────────
# sentence_transformers stub
# Some modules (utils.py) run SentenceTransformer() at import time, so this
# stub must be in sys.modules before any project module is imported.
# ─────────────────────────────────────────────────────────────────────────────
_st_instance = MagicMock()
_st_instance.encode.side_effect = lambda texts, **kw: (
    np.random.rand(len(texts), EMBED_DIM).astype(np.float32)
    if isinstance(texts, (list, tuple))
    else np.random.rand(1, EMBED_DIM).astype(np.float32)
)
_st_util = MagicMock()
# cos_sim must return something that supports .cpu().numpy()[0][0]
# (mirrors what a real sentence_transformers tensor looks like)
_cos_sim_result = MagicMock()
_cos_sim_result.cpu.return_value.numpy.return_value = np.array([[0.85]])
_st_util.cos_sim.return_value = _cos_sim_result

_st_mod = MagicMock()
_st_mod.SentenceTransformer = MagicMock(return_value=_st_instance)
_st_mod.util = _st_util
sys.modules["sentence_transformers"] = _st_mod
sys.modules["sentence_transformers.util"] = _st_util

# ─────────────────────────────────────────────────────────────────────────────
# langchain_classic stub (not declared in pyproject.toml; may not be installed)
# ─────────────────────────────────────────────────────────────────────────────
_hub_stub = MagicMock()
_hub_stub.pull.return_value = MagicMock()
_lc_classic = MagicMock()
_lc_classic.hub = _hub_stub
sys.modules["langchain_classic"] = _lc_classic
sys.modules["langchain_classic.hub"] = _hub_stub

# ─────────────────────────────────────────────────────────────────────────────
# LLM provider stubs — prevent real API/server calls
# Constructor kwargs are reflected on the returned instance so that assertions
# like `llm.model == "llama3.1"` work exactly as they would with a real object.
# ─────────────────────────────────────────────────────────────────────────────
def _make_ollama_instance(**kwargs):
    instance = MagicMock()
    _ollama_reply = AIMessage(content="mocked ollama answer")
    instance.invoke.return_value = _ollama_reply
    instance.return_value = _ollama_reply
    instance.model = kwargs.get("model", "llama3.1:8b")
    instance.temperature = kwargs.get("temperature", 0.0)
    instance.top_k = kwargs.get("top_k", None)
    instance.model_kwargs = {}
    return instance

_ollama_cls = MagicMock(side_effect=_make_ollama_instance)
_lo_mod = MagicMock()
_lo_mod.ChatOllama = _ollama_cls
sys.modules["langchain_ollama"] = _lo_mod


def _make_groq_instance(**kwargs):
    instance = MagicMock()
    _groq_reply = AIMessage(content="mocked groq answer")
    instance.invoke.return_value = _groq_reply
    instance.return_value = _groq_reply
    instance.model = kwargs.get("model", "llama-3.1-8b-instant")
    instance.temperature = kwargs.get("temperature", 0.0)
    return instance

_groq_cls = MagicMock(side_effect=_make_groq_instance)
_lg_mod = MagicMock()
_lg_mod.ChatGroq = _groq_cls
sys.modules["langchain_groq"] = _lg_mod

# ─────────────────────────────────────────────────────────────────────────────
# langchain_huggingface stub
# ─────────────────────────────────────────────────────────────────────────────
_hf_emb = MagicMock()
_hf_emb.embed_query.return_value = list(np.random.rand(EMBED_DIM).tolist())
_hf_emb.embed_documents.side_effect = lambda texts: [
    list(np.random.rand(EMBED_DIM).tolist()) for _ in texts
]
_hf_cls = MagicMock(return_value=_hf_emb)
_lhf_mod = MagicMock()
_lhf_mod.HuggingFaceEmbeddings = _hf_cls
sys.modules["langchain_huggingface"] = _lhf_mod


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_config():
    from pluto.rag.config import LLMProvider, RAGConfig
    return RAGConfig(
        texts=["France is a country in Europe. Its capital is Paris."],
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3.1:8b",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        chunk_size=200,
        llm_temperature=0.0,
        verbose=False,
    )


@pytest.fixture
def action_space():
    from pluto.rl.rl import RagAction
    return [
        RagAction(chunk_size=cs, llm_temperature=t)
        for cs in CHUNK_SIZES
        for t in TEMPERATURES
    ]


@pytest.fixture
def sample_items():
    return list(SAMPLE_QUESTIONS)


@pytest.fixture
def small_heatmap():
    return pd.DataFrame(
        [[0.5, 0.6, 0.4], [0.7, 0.8, 0.3], [0.6, 0.5, 0.9]],
        index=pd.Index(TEMPERATURES, name="temperature"),
        columns=pd.Index(CHUNK_SIZES, name="chunk_size"),
    )


@pytest.fixture
def synthetic_embeddings():
    """Two tight clusters separated by a clear gap; deterministic (seed=42)."""
    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=np.zeros(EMBED_DIM), scale=0.05, size=(6, EMBED_DIM))
    c1 = rng.normal(loc=np.ones(EMBED_DIM) * 5, scale=0.05, size=(6, EMBED_DIM))
    embeddings = np.vstack([c0, c1]).astype(np.float64)
    labels = np.array([0] * 6 + [1] * 6)
    return embeddings, labels


@pytest.fixture
def mock_reward_scorer():
    scorer = MagicMock()
    scorer.return_value = 0.75
    return scorer
