# datasets.py
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from datasets import load_dataset


class DataLoader(ABC):
    """Base class for dataset loaders."""

    @abstractmethod
    def load_random_article(self):
        raise NotImplementedError

    @abstractmethod
    def load_article_by_title(self, title: str):
        raise NotImplementedError


class SquadV2Loader(DataLoader):
    """Loader for SQuAD v2 dataset."""

    def __init__(self, split="train"):
        self.ds = load_dataset("squad_v2", split=split)
        self.articles = self._group_by_article()

    def _group_by_article(self):
        grouped = defaultdict(list)
        for ex in self.ds:
            if len(ex["answers"]["text"]) > 0:
                grouped[ex["title"]].append(ex)
        return grouped

    def load_random_article(self):
        """Returns a randomly selected article title, its items, and the combined long context."""
        title = random.choice(list(self.articles.keys()))
        items = self.articles[title]
        long_context = "\n\n".join(set(item["context"] for item in items))
        return title, items, long_context

    def load_article_by_title(self, title: str):
        """Loads article data by title."""
        if title not in self.articles:
            raise ValueError(f"Article with title '{title}' not found.")
        items = self.articles[title]
        long_context = "\n\n".join(set(item["context"] for item in items))
        return title, items, long_context
