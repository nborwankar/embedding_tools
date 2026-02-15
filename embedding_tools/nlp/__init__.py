"""NLP embedding wrappers with model registry, Matryoshka support, and migration toolkit."""

from .embedder import Embedder
from .extractors import JSONLExtractor, SQLiteExtractor, PgVectorExtractor

__all__ = ["Embedder", "JSONLExtractor", "SQLiteExtractor", "PgVectorExtractor"]
