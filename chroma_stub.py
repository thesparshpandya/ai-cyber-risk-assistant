"""
A minimal in-process stand-in for ChromaDB and sentence-transformers.

Used only by the test harness, so the retrieval integration (batched upsert,
metadata shape, query parsing, threshold filtering) can be exercised in
environments where torch cannot be installed. Similarity here is bag-of-words
cosine, not real embeddings, so distances are not representative of production.
"""

from __future__ import annotations

import math
import re
import sys
import types
from collections import Counter
from typing import Any, Dict, List, Optional


def _vector(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_distance(a: Counter, b: Counter) -> float:
    numerator = sum(count * b.get(token, 0) for token, count in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 1.0
    return 1.0 - numerator / (norm_a * norm_b)


class FakeCollection:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.upsert_calls = 0

    def count(self) -> int:
        return len(self.ids)

    def upsert(self, ids, documents, metadatas) -> None:
        self.upsert_calls += 1
        assert len(ids) == len(documents) == len(metadatas), "batch length mismatch"
        for identifier, document, metadata in zip(ids, documents, metadatas):
            assert isinstance(metadata, dict)
            for value in metadata.values():
                assert isinstance(value, (str, int, float, bool)), (
                    f"ChromaDB only accepts scalar metadata, got {type(value)}"
                )
            self.ids.append(identifier)
            self.documents.append(document)
            self.metadatas.append(metadata)

    def query(self, query_texts, n_results, include=None):
        query = _vector(query_texts[0])
        scored = sorted(
            ((_cosine_distance(query, _vector(doc)), i) for i, doc in enumerate(self.documents))
        )[:n_results]
        return {
            "ids": [[self.ids[i] for _, i in scored]],
            "documents": [[self.documents[i] for _, i in scored]],
            "metadatas": [[self.metadatas[i] for _, i in scored]],
            "distances": [[d for d, _ in scored]],
        }


class FakeClient:
    def __init__(self, path: Optional[str] = None, settings: Any = None) -> None:
        self.collections: Dict[str, FakeCollection] = {}

    def list_collections(self):
        # chromadb >=0.6 returns plain name strings; mirror that shape.
        return list(self.collections.keys())

    def get_collection(self, name: str, embedding_function: Any = None) -> FakeCollection:
        if name not in self.collections:
            raise ValueError(f"collection {name} does not exist")
        return self.collections[name]

    def get_or_create_collection(
        self, name: str, embedding_function: Any = None, metadata: Any = None
    ) -> FakeCollection:
        self.collections.setdefault(name, FakeCollection(name, metadata))
        return self.collections[name]


def install_stub() -> None:
    """Register the stub modules in sys.modules."""
    chromadb = types.ModuleType("chromadb")
    chromadb.__version__ = "stub"
    chromadb.PersistentClient = FakeClient

    config = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs: Any) -> None:
            pass

    config.Settings = Settings

    utils = types.ModuleType("chromadb.utils")
    embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")

    class SentenceTransformerEmbeddingFunction:
        def __init__(self, model_name: Optional[str] = None) -> None:
            self.model_name = model_name

    embedding_functions.SentenceTransformerEmbeddingFunction = (
        SentenceTransformerEmbeddingFunction
    )
    utils.embedding_functions = embedding_functions
    chromadb.config = config
    chromadb.utils = utils

    for name, module in (
        ("chromadb", chromadb),
        ("chromadb.config", config),
        ("chromadb.utils", utils),
        ("chromadb.utils.embedding_functions", embedding_functions),
    ):
        sys.modules[name] = module
