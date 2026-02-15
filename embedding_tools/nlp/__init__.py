"""NLP embedding wrappers with model registry, Matryoshka support, and migration toolkit."""

from .embedder import Embedder
from .extractors import JSONLExtractor, SQLiteExtractor, PgVectorExtractor
from .migration import MigrationComparator, MigrationReport

__all__ = [
    "Embedder",
    "JSONLExtractor",
    "SQLiteExtractor",
    "PgVectorExtractor",
    "MigrationComparator",
    "MigrationReport",
]
