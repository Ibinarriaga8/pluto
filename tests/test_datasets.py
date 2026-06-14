# test_datasets.py
"""Tests for SquadV2Loader (all HuggingFace dataset calls mocked)."""
from unittest.mock import patch

import pytest


def _make_squad_samples(n=10):
    """Build minimal SQuAD-style sample dicts with answers."""
    samples = []
    titles = ["France", "Germany", "Spain"]
    for i in range(n):
        title = titles[i % len(titles)]
        samples.append({
            "title": title,
            "context": f"Context text for {title} sample {i}.",
            "question": f"Q{i}?",
            "answers": {"text": [f"A{i}"], "answer_start": [0]},
        })
    return samples


def _make_squad_samples_no_answers(n=3):
    """Samples without answers (unanswerable questions in SQuAD v2)."""
    return [
        {
            "title": "Empty",
            "context": f"Context {i}",
            "question": f"Q{i}?",
            "answers": {"text": [], "answer_start": []},
        }
        for i in range(n)
    ]


class TestSquadV2Loader:
    @patch("pluto.eval.datasets.load_dataset")
    def test_instantiation_smoke(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(10)
        loader = SquadV2Loader(split="train")
        assert loader is not None

    @patch("pluto.eval.datasets.load_dataset")
    def test_load_dataset_called_once(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(6)
        SquadV2Loader(split="train")
        mock_load.assert_called_once_with("squad_v2", split="train")

    @patch("pluto.eval.datasets.load_dataset")
    def test_articles_grouped_by_title(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        # Three titles × 3 samples each
        assert "France" in loader.articles
        assert "Germany" in loader.articles
        assert "Spain" in loader.articles

    @patch("pluto.eval.datasets.load_dataset")
    def test_unanswerable_samples_excluded(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        samples = _make_squad_samples(6) + _make_squad_samples_no_answers(4)
        mock_load.return_value = samples
        loader = SquadV2Loader()
        all_items = [item for items in loader.articles.values() for item in items]
        # None of the stored items should have empty answers
        assert all(len(item["answers"]["text"]) > 0 for item in all_items)

    @patch("pluto.eval.datasets.load_dataset")
    def test_load_random_article_returns_three(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        title, items, long_context = loader.load_random_article()
        assert title is not None
        assert isinstance(items, list)
        assert isinstance(long_context, str)

    @patch("pluto.eval.datasets.load_dataset")
    def test_get_random_article_items_nonempty(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        _, items, _ = loader.load_random_article()
        assert len(items) > 0

    @patch("pluto.eval.datasets.load_dataset")
    def test_get_random_article_context_nonempty(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        _, _, long_context = loader.load_random_article()
        assert len(long_context) > 0

    @patch("pluto.eval.datasets.load_dataset")
    def test_load_by_title_returns_correct_data(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        title, items, context = loader.load_article_by_title("France")
        assert title == "France"
        assert all(item["title"] == "France" for item in items)

    @patch("pluto.eval.datasets.load_dataset")
    def test_load_by_unknown_title_raises(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        with pytest.raises(ValueError, match="NonExistentTitle"):
            loader.load_article_by_title("NonExistentTitle")

    @patch("pluto.eval.datasets.load_dataset")
    def test_long_context_combines_contexts(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        _, items, long_context = loader.load_article_by_title("France")
        unique_contexts = set(item["context"] for item in items)
        for ctx in unique_contexts:
            assert ctx in long_context

    @patch("pluto.eval.datasets.load_dataset")
    def test_articles_dict_populated(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(9)
        loader = SquadV2Loader()
        assert len(loader.articles) > 0

    @patch("pluto.eval.datasets.load_dataset")
    def test_split_parameter_forwarded(self, mock_load):
        from pluto.eval.datasets import SquadV2Loader
        mock_load.return_value = _make_squad_samples(3)
        SquadV2Loader(split="validation")
        mock_load.assert_called_once_with("squad_v2", split="validation")
