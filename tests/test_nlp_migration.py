"""Tests for MigrationComparator (unit tests, no model loading)."""

import pytest
import numpy as np

from embedding_tools.nlp.migration import MigrationComparator, MigrationReport


class TestMigrationComparatorUnit:
    def test_top_k_overlap_identical(self):
        """If both models return same results, overlap is 1.0."""
        comp = MigrationComparator.__new__(MigrationComparator)
        ids_a = ["d1", "d2", "d3"]
        ids_b = ["d1", "d2", "d3"]
        assert comp._overlap(ids_a, ids_b) == 1.0

    def test_top_k_overlap_disjoint(self):
        """If models return completely different results, overlap is 0.0."""
        comp = MigrationComparator.__new__(MigrationComparator)
        ids_a = ["d1", "d2", "d3"]
        ids_b = ["d4", "d5", "d6"]
        assert comp._overlap(ids_a, ids_b) == 0.0

    def test_top_k_overlap_partial(self):
        """50% overlap."""
        comp = MigrationComparator.__new__(MigrationComparator)
        ids_a = ["d1", "d2", "d3", "d4"]
        ids_b = ["d1", "d2", "d5", "d6"]
        assert comp._overlap(ids_a, ids_b) == 0.5

    def test_top_k_overlap_empty(self):
        """Empty lists have overlap 1.0 (vacuously true)."""
        comp = MigrationComparator.__new__(MigrationComparator)
        assert comp._overlap([], []) == 1.0

    def test_cosine_similarity(self):
        """Basic cosine similarity computation."""
        comp = MigrationComparator.__new__(MigrationComparator)
        q = np.array([1.0, 0.0])
        docs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sims = comp._cosine_similarity(q, docs)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)

    def test_cosine_similarity_normalized(self):
        """Cosine similarity is scale-invariant."""
        comp = MigrationComparator.__new__(MigrationComparator)
        q = np.array([3.0, 0.0])
        docs = np.array([[100.0, 0.0], [0.0, 5.0]])
        sims = comp._cosine_similarity(q, docs)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)

    def test_top_k_ids(self):
        """Top-K returns highest-scoring doc IDs in order."""
        comp = MigrationComparator.__new__(MigrationComparator)
        scores = np.array([0.1, 0.9, 0.5, 0.3])
        ids = ["a", "b", "c", "d"]
        top = comp._top_k_ids(scores, ids, 2)
        assert top == ["b", "c"]


class TestMigrationReport:
    def test_summary_go(self):
        report = MigrationReport(
            old_model="minilm-l6",
            new_model="nomic-v1.5",
            old_hf_name="all-MiniLM-L6-v2",
            new_hf_name="nomic-ai/nomic-embed-text-v1.5",
            num_documents=100,
            num_queries=5,
            top_k=10,
            per_query=[{"query": "test", "overlap": 0.7, "spearman": 0.8}],
            aggregate={
                "mean_overlap": 0.7,
                "min_overlap": 0.6,
                "mean_spearman": 0.8,
                "worst_query": "test",
                "worst_overlap": 0.6,
            },
        )
        s = report.summary()
        assert "GO" in s
        assert "INVESTIGATE" not in s

    def test_summary_investigate(self):
        report = MigrationReport(
            old_model="minilm-l6",
            new_model="nomic-v1.5",
            old_hf_name="all-MiniLM-L6-v2",
            new_hf_name="nomic-ai/nomic-embed-text-v1.5",
            num_documents=100,
            num_queries=5,
            top_k=10,
            per_query=[{"query": "test", "overlap": 0.3, "spearman": None}],
            aggregate={
                "mean_overlap": 0.3,
                "min_overlap": 0.2,
                "mean_spearman": None,
                "worst_query": "test",
                "worst_overlap": 0.2,
            },
        )
        s = report.summary()
        assert "INVESTIGATE" in s

    def test_save_and_load(self, tmp_path):
        report = MigrationReport(
            old_model="minilm-l6",
            new_model="nomic-v1.5",
            old_hf_name="all-MiniLM-L6-v2",
            new_hf_name="nomic-ai/nomic-embed-text-v1.5",
            num_documents=50,
            num_queries=3,
            top_k=10,
            per_query=[],
            aggregate={
                "mean_overlap": 0.6,
                "min_overlap": 0.5,
                "mean_spearman": None,
                "worst_query": "q1",
                "worst_overlap": 0.5,
            },
        )
        path = tmp_path / "report.json"
        report.save(str(path))
        import json

        data = json.loads(path.read_text())
        assert data["num_documents"] == 50
        assert data["old_model"] == "minilm-l6"


class TestPgvectorSQL:
    def test_generates_valid_sql(self):
        comp = MigrationComparator.__new__(MigrationComparator)
        sql = comp.generate_pgvector_comparison_sql(
            comparison_table="test_comparison", old_dim=384, new_dim=768
        )
        assert "CREATE TABLE test_comparison" in sql
        assert "vector(384)" in sql
        assert "vector(768)" in sql
        assert "FULL OUTER JOIN" in sql
