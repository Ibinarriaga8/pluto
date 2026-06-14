# test_rag_interface.py
"""Tests for InMemoryRAGInterface and ChromaRAGInterface."""
from unittest.mock import MagicMock, patch

# own modules
from pluto.rag.config import LLMProvider, RAGConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs):
    defaults = dict(
        texts=["France is a country. Its capital is Paris."],
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3.1:8b",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        chunk_size=100,
        llm_temperature=0.0,
        verbose=False,
    )
    defaults.update(kwargs)
    return RAGConfig(**defaults)


def _mock_rag_system(answer="Paris"):
    rag = MagicMock()
    rag.ask.return_value = answer
    return rag


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryRAGInterface
# ─────────────────────────────────────────────────────────────────────────────

class TestInMemoryRAGInterface:
    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_init_smoke(self, mock_init):
        from pluto.rag.rag_interface import InMemoryRAGInterface
        mock_init.return_value = _mock_rag_system()
        iface = InMemoryRAGInterface(_make_config())
        assert iface is not None

    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_ask_delegates_to_rag_system(self, mock_init):
        from pluto.rag.rag_interface import InMemoryRAGInterface
        mock_init.return_value = _mock_rag_system("Paris")
        iface = InMemoryRAGInterface(_make_config())
        result = iface.ask("What is the capital of France?")
        assert result == "Paris"

    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_ask_forwards_query(self, mock_init):
        from pluto.rag.rag_interface import InMemoryRAGInterface
        rag_mock = _mock_rag_system()
        mock_init.return_value = rag_mock
        iface = InMemoryRAGInterface(_make_config())
        iface.ask("My specific question")
        rag_mock.ask.assert_called_with("My specific question")

    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_config_is_stored(self, mock_init):
        from pluto.rag.rag_interface import InMemoryRAGInterface
        mock_init.return_value = _mock_rag_system()
        cfg = _make_config(chunk_size=512)
        iface = InMemoryRAGInterface(cfg)
        assert iface.config.chunk_size == 512

    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_embeddings_cache_reused(self, mock_init):
        """Two interfaces with the same model should share the cached embeddings."""
        from pluto.rag.rag_interface import _EMBEDDINGS_CACHE, InMemoryRAGInterface
        mock_init.return_value = _mock_rag_system()
        _EMBEDDINGS_CACHE.clear()

        iface1 = InMemoryRAGInterface(_make_config())
        iface2 = InMemoryRAGInterface(_make_config())
        assert iface1.embeddings is iface2.embeddings


# ─────────────────────────────────────────────────────────────────────────────
# ChromaRAGInterface
# ─────────────────────────────────────────────────────────────────────────────

class TestChromaRAGInterface:
    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_init_smoke(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        mock_init.return_value = _mock_rag_system()
        iface = ChromaRAGInterface(_make_config())
        assert iface is not None

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_ask_returns_answer(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        mock_init.return_value = _mock_rag_system("1991")
        iface = ChromaRAGInterface(_make_config())
        assert iface.ask("When was Python created?") == "1991"

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_sync_persist_directory_sets_path(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        mock_init.return_value = _mock_rag_system()
        cfg = _make_config(chunk_size=300)
        iface = ChromaRAGInterface(cfg)
        # Call _sync_persist_directory directly to verify its logic
        iface._sync_persist_directory()
        assert "cs300" in iface.config.persist_directory

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_update_temperature_smoke(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        rag_mock = _mock_rag_system()
        mock_init.return_value = rag_mock
        iface = ChromaRAGInterface(_make_config())
        iface.update_parameters(temperature=0.9)
        assert iface.config.llm_temperature == 0.9

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_update_top_k_smoke(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        rag_mock = _mock_rag_system()
        rag_mock.retriever = MagicMock()
        mock_init.return_value = rag_mock
        iface = ChromaRAGInterface(_make_config())
        iface.update_parameters(top_k=8)
        assert iface.config.top_k == 8

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_update_chunk_size_triggers_reinit(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        rag_mock = _mock_rag_system()
        mock_init.return_value = rag_mock
        iface = ChromaRAGInterface(_make_config(chunk_size=100))
        call_count_before = mock_init.call_count
        iface.update_parameters(chunk_size=500)
        assert mock_init.call_count > call_count_before

    @patch("pluto.rag.rag_interface.ChromaRAGInterface._initialize_system")
    def test_update_same_chunk_size_no_reinit(self, mock_init):
        from pluto.rag.rag_interface import ChromaRAGInterface
        mock_init.return_value = _mock_rag_system()
        iface = ChromaRAGInterface(_make_config(chunk_size=100))
        count = mock_init.call_count
        iface.update_parameters(chunk_size=100)
        assert mock_init.call_count == count


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings cache (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingsCache:
    def test_cache_is_dict(self):
        from pluto.rag.rag_interface import _EMBEDDINGS_CACHE
        assert isinstance(_EMBEDDINGS_CACHE, dict)

    @patch("pluto.rag.rag_interface.InMemoryRAGInterface._initialize_system")
    def test_get_cached_embeddings_same_object(self, mock_init):
        from pluto.rag.rag_interface import _EMBEDDINGS_CACHE, InMemoryRAGInterface
        mock_init.return_value = _mock_rag_system()
        model = "sentence-transformers/all-mpnet-base-v2"
        _EMBEDDINGS_CACHE.clear()

        e1 = InMemoryRAGInterface._get_cached_embeddings(model)
        e2 = InMemoryRAGInterface._get_cached_embeddings(model)
        assert e1 is e2
