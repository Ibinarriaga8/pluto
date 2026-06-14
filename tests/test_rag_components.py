# test_rag_components.py
"""Tests for the low-level RAG building blocks (Loader, Indexer, Retriever, Generator, RAG)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_docs(n=3):
    return [Document(page_content=f"doc {i}", metadata={"source": f"s{i}"}) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# ConfigurableLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurableLoader:
    def test_load_from_texts(self):
        from pluto.rag.rag import ConfigurableLoader
        loader = ConfigurableLoader(texts=["Hello world", "Foo bar"])
        docs = loader.load()
        assert len(docs) == 2

    def test_load_text_content(self):
        from pluto.rag.rag import ConfigurableLoader
        loader = ConfigurableLoader(texts=["Alpha", "Beta"])
        docs = loader.load()
        contents = [d.page_content for d in docs]
        assert "Alpha" in contents
        assert "Beta" in contents

    def test_load_empty_texts_returns_empty(self):
        from pluto.rag.rag import ConfigurableLoader
        loader = ConfigurableLoader(texts=[])
        docs = loader.load()
        assert docs == []

    def test_load_single_text(self):
        from pluto.rag.rag import ConfigurableLoader
        loader = ConfigurableLoader(texts=["Only one"])
        docs = loader.load()
        assert len(docs) == 1

    @patch("pluto.rag.rag.WebBaseLoader")
    def test_load_from_url_calls_web_loader(self, mock_web):
        from pluto.rag.rag import ConfigurableLoader
        mock_web.return_value.load.return_value = _make_docs(2)
        loader = ConfigurableLoader(urls=["http://example.com"])
        docs = loader.load()
        assert len(docs) == 2
        # __init__ + load() both instantiate WebBaseLoader
        assert mock_web.call_count >= 1

    @patch("pluto.rag.rag.WebBaseLoader")
    def test_load_url_and_texts_combined(self, mock_web):
        from pluto.rag.rag import ConfigurableLoader
        mock_web.return_value.load.return_value = _make_docs(1)
        loader = ConfigurableLoader(urls=["http://example.com"], texts=["Extra text"])
        docs = loader.load()
        assert len(docs) == 2

    @patch("pluto.rag.rag.WebBaseLoader")
    def test_url_load_failure_does_not_raise_with_texts(self, mock_web):
        from pluto.rag.rag import ConfigurableLoader
        mock_web.return_value.load.side_effect = Exception("network error")
        loader = ConfigurableLoader(urls=["http://bad-url.invalid"], texts=["fallback"])
        docs = loader.load()
        assert len(docs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryIndexer
# ─────────────────────────────────────────────────────────────────────────────

class TestInMemoryIndexer:
    def _make_indexer(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from pluto.rag.rag import InMemoryIndexer
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [
            np.random.rand(16).tolist() for _ in range(10)
        ]
        return InMemoryIndexer(embeddings_model=embeddings, text_splitter=splitter)

    @patch("pluto.rag.rag.InMemoryVectorStore")
    def test_index_returns_vector_store(self, mock_vs):
        mock_vs.return_value = MagicMock()
        indexer = self._make_indexer()
        result = indexer.index_documents(_make_docs(2))
        assert result is not None

    @patch("pluto.rag.rag.InMemoryVectorStore")
    def test_index_calls_add_documents(self, mock_vs):
        vs_instance = MagicMock()
        mock_vs.return_value = vs_instance
        indexer = self._make_indexer()
        indexer.index_documents(_make_docs(3))
        vs_instance.add_documents.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryRetriever
# ─────────────────────────────────────────────────────────────────────────────

class TestInMemoryRetriever:
    def test_get_retriever_returns_runnable(self):
        from langchain_community.vectorstores import InMemoryVectorStore

        from pluto.rag.rag import InMemoryRetriever

        mock_vs = MagicMock(spec=InMemoryVectorStore)
        mock_vs.as_retriever.return_value = MagicMock()

        retriever = InMemoryRetriever(top_k=3)
        result = retriever.get_retriever(mock_vs)
        assert result is not None
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})

    def test_wrong_vector_store_type_raises(self):
        from pluto.rag.rag import InMemoryRetriever
        retriever = InMemoryRetriever(top_k=3)
        with pytest.raises(ValueError, match="InMemoryRetriever"):
            retriever.get_retriever(object())


# ─────────────────────────────────────────────────────────────────────────────
# LangChainGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestLangChainGenerator:
    def test_get_chain_returns_runnable(self):
        from pluto.rag.rag import LangChainGenerator
        llm = MagicMock()
        prompt = MagicMock()
        retriever = MagicMock()
        gen = LangChainGenerator(llm=llm, prompt_template=prompt)
        chain = gen.get_chain(retriever)
        assert chain is not None


# ─────────────────────────────────────────────────────────────────────────────
# RAG (composition class)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAG:
    def _make_rag(self, answer="Paris"):
        from pluto.rag.rag import RAG
        loader = MagicMock()
        loader.load.return_value = _make_docs(2)

        vs = MagicMock()
        indexer = MagicMock()
        indexer.index_documents.return_value = vs

        fake_retriever_runnable = MagicMock()
        fake_retriever_runnable.invoke.return_value = _make_docs(2)
        retriever = MagicMock()
        retriever.get_retriever.return_value = fake_retriever_runnable

        chain = MagicMock()
        chain.invoke.return_value = answer
        generator = MagicMock()
        generator.get_chain.return_value = chain

        rag = RAG(loader=loader, indexer=indexer, retriever=retriever, generator=generator)
        return rag

    def test_setup_pipeline_smoke(self):
        rag = self._make_rag()
        rag.setup_pipeline()

    def test_ask_before_setup_raises(self):
        rag = self._make_rag()
        with pytest.raises(ValueError):
            rag.ask("What is the capital?")

    def test_ask_after_setup_returns_answer(self):
        rag = self._make_rag(answer="Paris")
        rag.setup_pipeline()
        result = rag.ask("What is the capital of France?")
        assert result == "Paris"

    def test_ask_calls_chain_invoke(self):
        rag = self._make_rag()
        rag.setup_pipeline()
        rag.ask("Some question")
        rag.rag_chain.invoke.assert_called()
