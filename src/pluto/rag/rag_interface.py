# rag_interface.py
from __future__ import annotations

import abc
import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

_DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    )),
])
from langchain_text_splitters import RecursiveCharacterTextSplitter

# own modules
from pluto.rag.config import RAGConfig
from pluto.rag.llm_loader import LLMFactory
from pluto.rag.rag import (
    RAG,
    ChromaIndexer,
    ChromaRetriever,
    ConfigurableLoader,
    InMemoryIndexer,
    InMemoryRetriever,
    LangChainGenerator,
)

logger = logging.getLogger(__name__)

# Shared cache to prevent redundant VRAM usage across different interface instances.
_EMBEDDINGS_CACHE: dict[str, HuggingFaceEmbeddings] = {}


class BaseRAGInterface(abc.ABC):
    """Abstract base for RAG system wrappers over different vector store backends."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embeddings = self._get_cached_embeddings(config.embedding_model_name)
        self.llm = LLMFactory.create_llm(config)
        self.rag_system = self._initialize_system()

    @abc.abstractmethod
    def _initialize_system(self) -> RAG:
        pass

    @staticmethod
    def _get_cached_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        if model_name not in _EMBEDDINGS_CACHE:
            logger.info("Loading embedding model: %s", model_name)
            _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        else:
            logger.debug("Using cached embedding model: %s", model_name)
        return _EMBEDDINGS_CACHE[model_name]

    def ask(self, query: str) -> str:
        return self.rag_system.ask(query)


class InMemoryRAGInterface(BaseRAGInterface):
    """RAG interface backed by an in-memory vector store. Ideal for rapid iterations."""

    def _initialize_system(self) -> RAG:
        logger.info("Initializing in-memory RAG interface (%s).", self.config.llm_provider.value)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        prompt = self.config.custom_prompt_template or _DEFAULT_RAG_PROMPT
        loader = ConfigurableLoader(urls=self.config.urls, texts=self.config.texts)

        indexer = InMemoryIndexer(embeddings_model=self.embeddings, text_splitter=splitter)
        retriever = InMemoryRetriever(top_k=self.config.top_k)
        generator = LangChainGenerator(llm=self.llm, prompt_template=prompt)

        rag = RAG(loader=loader, indexer=indexer, retriever=retriever, generator=generator, verbose=self.config.verbose)
        rag.setup_pipeline()
        return rag


class ChromaRAGInterface(BaseRAGInterface):
    """RAG interface backed by a persistent ChromaDB store."""

    def _initialize_system(self) -> RAG:
        self._sync_persist_directory()

        logger.info("Initializing persistent Chroma RAG interface at %s.", self.config.persist_directory)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        prompt = self.config.custom_prompt_template or _DEFAULT_RAG_PROMPT
        loader = ConfigurableLoader(urls=self.config.urls, texts=self.config.texts)

        indexer = ChromaIndexer(
            embeddings_model=self.embeddings,
            text_splitter=splitter,
            persist_directory=self.config.persist_directory,
        )
        retriever = ChromaRetriever(
            embeddings_model=self.embeddings,
            persist_directory=self.config.persist_directory,
            top_k=self.config.top_k,
        )
        generator = LangChainGenerator(llm=self.llm, prompt_template=prompt)

        rag = RAG(loader=loader, indexer=indexer, retriever=retriever, generator=generator, verbose=self.config.verbose)
        rag.setup_pipeline()
        return rag

    def _sync_persist_directory(self) -> None:
        folder_name = f"db_cs{self.config.chunk_size}_ol{self.config.chunk_overlap}"
        self.config.persist_directory = os.path.join(self.config.base_path, folder_name)

    def update_parameters(
        self,
        llm_top_k: int | None = None,
        top_k: int | None = None,
        chunk_size: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Updates parameters and triggers re-indexing only if chunk_size changes."""
        if llm_top_k is not None:
            self.config.llm_top_k = llm_top_k
            if hasattr(self.llm, "top_k"):
                self.llm.top_k = llm_top_k
            else:
                self.llm.model_kwargs = {**self.llm.model_kwargs, "top_k": llm_top_k}

        if top_k is not None:
            self.config.top_k = top_k
            self.rag_system.retriever.top_k = top_k

        if temperature is not None:
            self.config.llm_temperature = temperature
            self.llm.temperature = temperature

        if chunk_size is not None and self.config.chunk_size != chunk_size:
            logger.info("chunk_size changed (%d → %d), re-indexing...", self.config.chunk_size, chunk_size)
            self.config.chunk_size = chunk_size
            # Keep overlap at ~20 % of chunk size so it is always < chunk_size.
            # The old formula (min(chunk_size, old_overlap)) gave overlap == chunk_size
            # for small chunks (e.g. cs=100, default_ol=200 → ol=100 = 100 % overlap).
            self.config.chunk_overlap = max(0, chunk_size // 5)
            self.rag_system = self._initialize_system()
