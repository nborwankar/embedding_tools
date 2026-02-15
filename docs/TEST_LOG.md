# Test Run Log — Embedding Unification

> **Future data mining note:** This log uses a consistent tabular format per run
> so entries can be imported into SQLite/PostgreSQL for trend analysis (test count
> over time, duration regressions, failure rate by phase, etc.). Each run is one
> logical row: `(datetime, project, phase, task, command, passed, failed, skipped,
> deselected, duration_sec, context, commit, raw_output)`.

> **Raw output:** Full pytest output is saved in `docs/test_runs/`.
> Naming convention: `YYYY-MM-DD_p{phase}_t{task}_{description}.txt`

## Phase 0: Embedder + Migration Toolkit in embedding_tools

| datetime | project | phase | task | command | passed | failed | skipped | deselected | duration_sec | context | commit | raw_output |
|----------|---------|-------|------|---------|--------|--------|---------|------------|--------------|---------|--------|------------|
| 2026-02-14T16:00 | embedding_tools | 0 | 0.2 | `pytest tests/test_nlp_embedder.py -v` | 10 | 0 | 0 | 0 | 0.79 | Embedder class with model registry. Tests verify registry contents, config, Matryoshka, error handling. No model downloads. | 5cb14ae | [raw](test_runs/2026-02-14_p0_t0.2_embedder.txt) |
| 2026-02-14T16:05 | embedding_tools | 0 | 0.3 | `pytest tests/test_nlp_extractors.py -v` | 8 | 0 | 0 | 0 | 0.86 | JSONLExtractor, SQLiteExtractor, PgVectorExtractor. Tests use tmp_path with real SQLite DBs and JSONL files. | e1159cf | [raw](test_runs/2026-02-14_p0_t0.3_extractors.txt) |
| 2026-02-14T16:10 | embedding_tools | 0 | 0.4 | `pytest tests/test_nlp_migration.py -v` | 11 | 0 | 0 | 0 | 0.61 | MigrationComparator overlap, cosine sim, top-K, Spearman. MigrationReport summary/save. PgVector SQL gen. | 4ad4afe | [raw](test_runs/2026-02-14_p0_t0.4_migration.txt) |
| 2026-02-14T16:20 | embedding_tools | 0 | 0.5 | `pytest tests/test_nlp_integration.py -v -m slow` | 6 | 0 | 0 | 0 | 28.74 | End-to-end: nomic-v1.5, jina-code-v2, minilm-l6 load+encode. Full MigrationComparator with SQLite. Required einops fix. | f211d9c | [raw](test_runs/2026-02-14_p0_t0.5_integration.txt) |
| 2026-02-14T16:25 | embedding_tools | 0 | 0.6 | `pytest tests/ -v -m "not slow"` | 81 | 0 | 23 | 6 | 5.36 | Full suite after v0.2.0 bump. 23 skipped = JAX not installed. 6 deselected = slow marker. No regressions. | 1e7a279 | [raw](test_runs/2026-02-14_p0_t0.6_full_suite.txt) |

**Phase 0 complete.** All 87 tests pass (81 fast + 6 slow).
