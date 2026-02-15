"""Migration comparison toolkit for embedding model transitions.

Provides apples-to-apples comparison between old and new embedding models
using the project's actual source text and queries. Produces a JSON report
with per-query top-K overlap and Spearman rank correlation.

Usage:
    from embedding_tools.nlp import Embedder, PgVectorExtractor
    from embedding_tools.nlp.migration import MigrationComparator

    source = PgVectorExtractor(
        dbname="redditmath", table="topic_chunks",
        text_col="content", id_col="id",
    )
    comparator = MigrationComparator(
        source=source,
        old_model="minilm-l6",
        new_model="nomic-v1.5",
        queries=["what is a prime number", "eigenvalue decomposition"],
    )
    report = comparator.run(top_k=10)
    report.save("migration_report.json")
    print(report.summary())
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class MigrationReport:
    """Results of a migration comparison."""

    old_model: str
    new_model: str
    old_hf_name: str
    new_hf_name: str
    num_documents: int
    num_queries: int
    top_k: int
    per_query: List[dict]
    aggregate: dict

    def summary(self) -> str:
        lines = [
            f"Migration comparison: {self.old_model} -> {self.new_model}",
            f"  Documents: {self.num_documents}, Queries: {self.num_queries}",
            f"  Mean top-{self.top_k} overlap: {self.aggregate['mean_overlap']:.3f}",
            f"  Min overlap:  {self.aggregate['min_overlap']:.3f}",
        ]
        if self.aggregate.get("mean_spearman") is not None:
            lines.append(f"  Mean Spearman: {self.aggregate['mean_spearman']:.3f}")
        lines.append(
            f"  Worst query: '{self.aggregate['worst_query']}' "
            f"(overlap={self.aggregate['worst_overlap']:.3f})"
        )
        go = self.aggregate["mean_overlap"] >= 0.5
        lines.append(f"  Recommendation: {'GO' if go else 'INVESTIGATE'}")
        return "\n".join(lines)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "old_model": self.old_model,
                    "new_model": self.new_model,
                    "old_hf_name": self.old_hf_name,
                    "new_hf_name": self.new_hf_name,
                    "num_documents": self.num_documents,
                    "num_queries": self.num_queries,
                    "top_k": self.top_k,
                    "per_query": self.per_query,
                    "aggregate": self.aggregate,
                },
                f,
                indent=2,
            )


class MigrationComparator:
    """Compare retrieval quality between old and new embedding models.

    Uses the project's actual source text (from any extractor) and
    representative queries to compute top-K overlap and rank correlation.
    """

    def __init__(self, source, old_model: str, new_model: str, queries: List[str]):
        """
        Args:
            source: Any extractor (PgVectorExtractor, SQLiteExtractor, etc.)
                    Must have .extract() yielding {"id": str, "text": str}.
            old_model: Registry key for current model (e.g., "minilm-l6").
            new_model: Registry key for target model (e.g., "nomic-v1.5").
            queries: List of representative search query strings.
        """
        self.source = source
        self.old_model_key = old_model
        self.new_model_key = new_model
        self.queries = queries

    def _cosine_similarity(self, query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between query and all docs."""
        q_norm = query_vec / np.linalg.norm(query_vec)
        d_norms = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        return d_norms @ q_norm

    def _top_k_ids(self, scores: np.ndarray, doc_ids: list, k: int) -> list:
        indices = np.argsort(scores)[::-1][:k]
        return [doc_ids[i] for i in indices]

    def _overlap(self, ids_a: list, ids_b: list) -> float:
        k = max(len(ids_a), len(ids_b))
        if k == 0:
            return 1.0
        return len(set(ids_a) & set(ids_b)) / k

    def run(self, top_k: int = 10) -> MigrationReport:
        """Run the comparison. Returns a MigrationReport."""
        from .embedder import Embedder

        # Extract source text
        docs = list(self.source.extract())
        doc_ids = [d["id"] for d in docs]
        doc_texts = [d["text"] for d in docs]

        # Embed all docs with both models
        old_embedder = Embedder(self.old_model_key)
        new_embedder = Embedder(self.new_model_key)

        old_doc_embs = old_embedder.encode(doc_texts)
        new_doc_embs = new_embedder.encode(doc_texts)

        # Compare per query
        results = []
        for query_text in self.queries:
            old_q = old_embedder.encode_single(query_text)
            new_q = new_embedder.encode_single(query_text)

            old_scores = self._cosine_similarity(old_q, old_doc_embs)
            new_scores = self._cosine_similarity(new_q, new_doc_embs)

            old_top = self._top_k_ids(old_scores, doc_ids, top_k)
            new_top = self._top_k_ids(new_scores, doc_ids, top_k)

            overlap = self._overlap(old_top, new_top)

            # Spearman on shared docs
            shared = list(set(old_top) & set(new_top))
            sp = None
            if len(shared) >= 2:
                from scipy.stats import spearmanr

                old_ranks = [old_top.index(d) for d in shared]
                new_ranks = [new_top.index(d) for d in shared]
                sp = float(spearmanr(old_ranks, new_ranks).statistic)

            results.append(
                {
                    "query": query_text,
                    "overlap": overlap,
                    "spearman": sp,
                    "old_top": old_top,
                    "new_top": new_top,
                }
            )

        # Aggregate
        overlaps = [r["overlap"] for r in results]
        spearman_vals = [r["spearman"] for r in results if r["spearman"] is not None]

        aggregate = {
            "mean_overlap": float(np.mean(overlaps)),
            "min_overlap": float(np.min(overlaps)),
            "mean_spearman": float(np.mean(spearman_vals)) if spearman_vals else None,
            "worst_query": results[int(np.argmin(overlaps))]["query"],
            "worst_overlap": float(np.min(overlaps)),
        }

        return MigrationReport(
            old_model=self.old_model_key,
            new_model=self.new_model_key,
            old_hf_name=old_embedder.model_name,
            new_hf_name=new_embedder.model_name,
            num_documents=len(docs),
            num_queries=len(self.queries),
            top_k=top_k,
            per_query=results,
            aggregate=aggregate,
        )

    def generate_pgvector_comparison_sql(
        self,
        comparison_table: str = "embedding_comparison",
        old_dim: int = 384,
        new_dim: int = 768,
    ) -> str:
        """Generate SQL for a dual-column comparison table in PostgreSQL."""
        return f"""-- Dual-column comparison table (temporary, for migration validation)
CREATE TABLE {comparison_table} (
    id          SERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL,
    chunk_text  TEXT NOT NULL,
    emb_old     vector({old_dim}),
    emb_new     vector({new_dim})
);

-- After populating, compare top-10 for any query with:
-- (pass query embeddings as $1::vector({old_dim}) and $2::vector({new_dim}))
WITH ranked_old AS (
    SELECT source_id, LEFT(chunk_text, 80) AS text_preview,
           1 - (emb_old <=> $1::vector({old_dim})) AS score,
           ROW_NUMBER() OVER (ORDER BY emb_old <=> $1::vector({old_dim})) AS rank
    FROM {comparison_table} WHERE emb_old IS NOT NULL LIMIT 10
),
ranked_new AS (
    SELECT source_id, LEFT(chunk_text, 80) AS text_preview,
           1 - (emb_new <=> $2::vector({new_dim})) AS score,
           ROW_NUMBER() OVER (ORDER BY emb_new <=> $2::vector({new_dim})) AS rank
    FROM {comparison_table} WHERE emb_new IS NOT NULL LIMIT 10
)
SELECT COALESCE(o.source_id, n.source_id) AS doc_id,
       COALESCE(o.text_preview, n.text_preview) AS text,
       o.rank AS old_rank, o.score AS old_score,
       n.rank AS new_rank, n.score AS new_score
FROM ranked_old o FULL OUTER JOIN ranked_new n ON o.source_id = n.source_id
ORDER BY COALESCE(o.rank, 999), COALESCE(n.rank, 999);
"""
