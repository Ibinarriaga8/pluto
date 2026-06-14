# rag.py
import abc
import logging
from typing import Any

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

logger = logging.getLogger(__name__)


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> list[Document]:
        pass


class BaseIndexer(abc.ABC):
    @abc.abstractmethod
    def index_documents(self, docs: list[Document]) -> Any:
        pass


class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def get_retriever(self, vector_store: Any) -> Runnable:
        pass


class BaseGenerator(abc.ABC):
    @abc.abstractmethod
    def get_chain(self, retriever: Runnable) -> Runnable:
        pass


class ConfigurableLoader(BaseLoader):
    """Loads documents from URLs and/or raw texts."""

    def __init__(self, urls=None, texts: list[str] | None = None, verbose: bool = False):
        self.urls = urls or []
        self.texts = texts or []
        self.verbose = verbose

    def load(self) -> list[Document]:
        docs = []

        if self.urls:
            logger.debug("Loading documents from URLs: %s", self.urls)
            try:
                url_loader = WebBaseLoader(web_paths=self.urls)
                url_docs = url_loader.load()
                docs.extend(url_docs)
                logger.debug("Loaded %d URL documents.", len(url_docs))
            except Exception as e:
                logger.warning("URL loading failed: %s", e)

        if self.texts:
            logger.debug("Loading %d in-memory text documents...", len(self.texts))
            text_docs = [
                Document(page_content=t, metadata={"source": f"text_{i}"})
                for i, t in enumerate(self.texts)
            ]
            docs.extend(text_docs)

        return docs


class ChromaIndexer(BaseIndexer):
    """Indexes documents into a persistent ChromaDB store."""

    def __init__(self, embeddings_model, text_splitter, persist_directory: str, verbose: bool = False):
        self.embeddings = embeddings_model
        self.text_splitter = text_splitter
        self.persist_directory = persist_directory
        self.verbose = verbose
        self.vector_store = None

    def index_documents(self, docs: list[Document]) -> Chroma:
        # Delete any existing collection through ChromaDB's own API so stale
        # data from previous runs doesn't accumulate. Using delete_collection()
        # rather than shutil.rmtree avoids SQLITE_READONLY_DBMOVED errors that
        # occur when another ChromaRAGInterface instance still holds an open
        # connection to the same persist directory.
        _existing = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        _existing.delete_collection()
        del _existing

        logger.debug("Initializing persistent vector store at %s...", self.persist_directory)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        if not docs:
            logger.warning("No documents provided to index. Proceeding with empty/existing store.")
            return self.vector_store

        logger.debug("Splitting %d documents...", len(docs))
        split_docs = self.text_splitter.split_documents(docs)

        if split_docs:
            logger.debug("Adding %d chunks to the vector store...", len(split_docs))
            batch_size = 5000
            for i in range(0, len(split_docs), batch_size):
                self.vector_store.add_documents(split_docs[i : i + batch_size])
            logger.debug("Indexing complete.")

        return self.vector_store


class InMemoryIndexer(BaseIndexer):
    """Indexes documents into an in-memory vector store."""

    def __init__(self, embeddings_model, text_splitter, verbose: bool = False):
        self.embeddings = embeddings_model
        self.text_splitter = text_splitter
        self.verbose = verbose

    def index_documents(self, docs: list[Document]) -> InMemoryVectorStore:
        logger.debug("Splitting %d documents...", len(docs))
        split_docs = self.text_splitter.split_documents(docs)

        logger.debug("Creating in-memory vector store...")
        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(documents=split_docs)

        logger.debug("Indexing complete. %d chunks added to memory.", len(split_docs))
        return vector_store


class ChromaRetriever(BaseRetriever):
    """Retrieves from a persistent ChromaDB store."""

    def __init__(self, embeddings_model, persist_directory: str, top_k: int, verbose: bool = False):
        self.persist_directory = persist_directory
        self.embeddings = embeddings_model
        self.top_k = top_k
        self.verbose = verbose

    def get_retriever(self, _vector_store: Any = None) -> Runnable:
        """
        Loads the persistent vector store from disk.
        The vector_store argument is unused; ChromaRetriever loads from disk directly.
        """
        logger.debug("Loading persistent vector store from %s...", self.persist_directory)
        try:
            vector_store_from_disk = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return vector_store_from_disk.as_retriever(
                search_kwargs={"k": self.top_k}
            )
        except Exception as e:
            logger.error("Error loading vector store: %s", e)
            raise ValueError("Could not load persistent vector store. Run indexing first.") from e


class InMemoryRetriever(BaseRetriever):
    """Retrieves from an in-memory vector store."""

    def __init__(self, top_k: int, verbose: bool = False):
        self.top_k = top_k
        self.verbose = verbose

    def get_retriever(self, vector_store: Any) -> Runnable:
        if not isinstance(vector_store, InMemoryVectorStore):
            raise ValueError(
                "InMemoryRetriever requires an InMemoryVectorStore object. "
                "Please run the InMemoryIndexer first."
            )
        logger.debug("Using in-memory vector store for retrieval...")
        return vector_store.as_retriever(search_kwargs={"k": self.top_k})


class LangChainGenerator(BaseGenerator):
    """Generates responses using a LangChain LCEL chain."""

    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.prompt = prompt_template

    def get_chain(self, retriever: Runnable) -> Runnable:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain


class RAG:
    """Composes Loader, Indexer, Retriever, and Generator into a single pipeline."""

    def __init__(
        self,
        loader: BaseLoader,
        indexer: BaseIndexer,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        verbose: bool = False
    ):
        self.loader = loader
        self.indexer = indexer
        self.retriever = retriever
        self.generator = generator
        self.vector_store = None
        self.rag_chain = None
        self.verbose = verbose

    def setup_pipeline(self) -> None:
        docs = self.loader.load()

        logger.info("Indexing documents...")
        self.vector_store = self.indexer.index_documents(docs)

        retriever_runnable = self.retriever.get_retriever(self.vector_store)
        self.rag_chain = self.generator.get_chain(retriever_runnable)
        logger.info("RAG pipeline ready.")

    def ask(self, query: str) -> str:
        if not self.rag_chain:
            raise ValueError("RAG chain not loaded. Call .setup_pipeline() first.")

        logger.debug("Query: %s", query)
        response = self.rag_chain.invoke(query)
        logger.debug("Answer: %s", response)
        return response
