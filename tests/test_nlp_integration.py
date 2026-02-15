"""Integration tests that download models and run full comparisons."""

import json
import sqlite3

import numpy as np
import pytest

from embedding_tools.nlp import (
    Embedder,
    SQLiteExtractor,
    JSONLExtractor,
    MigrationComparator,
)

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform input data into predictions.",
    "PostgreSQL supports vector similarity search with pgvector.",
]

SAMPLE_CODE = [
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "class DatabaseConnection: def __init__(self, host, port): self.host = host",
    "import numpy as np; arr = np.array([1, 2, 3])",
]


@pytest.mark.slow
class TestEmbedderIntegration:
    def test_nomic_loads_and_encodes(self):
        e = Embedder("nomic-v1.5")
        result = e.encode(SAMPLE_TEXTS)
        assert result.shape == (3, 768)
        assert result.dtype == np.float32

    def test_nomic_matryoshka_256(self):
        e = Embedder("nomic-v1.5", truncate_dim=256)
        result = e.encode(SAMPLE_TEXTS)
        assert result.shape == (3, 256)

    def test_jina_code_loads_and_encodes(self):
        e = Embedder("jina-code-v2")
        result = e.encode(SAMPLE_CODE)
        assert result.shape == (3, 768)

    def test_minilm_legacy_loads(self):
        e = Embedder("minilm-l6")
        result = e.encode(SAMPLE_TEXTS)
        assert result.shape == (3, 384)

    def test_cosine_similarity_makes_sense(self):
        e = Embedder("nomic-v1.5")
        embs = e.encode(
            [
                "cats are great pets",
                "dogs are wonderful companions",
                "the stock market crashed today",
            ]
        )
        from numpy.linalg import norm

        def cos(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))

        assert cos(embs[0], embs[1]) > cos(embs[0], embs[2])


@pytest.mark.slow
class TestMigrationComparatorIntegration:
    def test_full_comparison_with_sqlite(self, tmp_path):
        """End-to-end: create SQLite DB, extract, compare, get report."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        docs = [
            "Linear algebra studies vector spaces and linear mappings.",
            "Calculus deals with derivatives and integrals.",
            "Topology studies properties preserved under continuous deformations.",
            "Statistics analyzes data to find patterns and make predictions.",
            "Number theory studies properties of integers and prime numbers.",
        ]
        for i, text in enumerate(docs):
            conn.execute("INSERT INTO docs VALUES (?, ?)", (i, text))
        conn.commit()
        conn.close()

        source = SQLiteExtractor(str(db), table="docs", id_col="id", text_col="content")
        comparator = MigrationComparator(
            source=source,
            old_model="minilm-l6",
            new_model="nomic-v1.5",
            queries=["What is a vector space?", "How do derivatives work?"],
        )
        report = comparator.run(top_k=3)

        assert report.num_documents == 5
        assert report.num_queries == 2
        assert 0 <= report.aggregate["mean_overlap"] <= 1
        assert report.aggregate["worst_query"] is not None

        # Save and reload
        out = tmp_path / "report.json"
        report.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["num_documents"] == 5

        # Summary is printable
        s = report.summary()
        assert "GO" in s or "INVESTIGATE" in s
