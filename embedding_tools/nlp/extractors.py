"""Source text extractors for embedding migration.

Extractors pull (id, text) pairs from wherever embeddings are stored.
Used by MigrationComparator to do apples-to-apples model comparison
on a project's actual data.

Supported backends:
  - JSONLExtractor:    reads from .jsonl files
  - SQLiteExtractor:   reads from SQLite tables (e.g., strictRAG cartridges)
  - PgVectorExtractor: reads from PostgreSQL+pgvector tables

Each extractor yields {"id": str, "text": str} dicts.
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterator, Optional


class JSONLExtractor:
    """Extract source text from a JSONL file."""

    def __init__(self, path: str, id_col: str = "id", text_col: str = "text"):
        self.path = path
        self.id_col = id_col
        self.text_col = text_col

    def extract(self) -> Iterator[dict]:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield {"id": str(row[self.id_col]), "text": row[self.text_col]}

    def count(self) -> int:
        return sum(1 for _ in self.extract())

    def save_jsonl(self, output_path: str):
        """Save extracted data to a JSONL file (for use as comparison input)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for doc in self.extract():
                f.write(json.dumps(doc) + "\n")


class SQLiteExtractor:
    """Extract source text from a SQLite table."""

    def __init__(
        self,
        db_path: str,
        table: str,
        id_col: str = "id",
        text_col: str = "content",
        where: Optional[str] = None,
    ):
        self.db_path = db_path
        self.table = table
        self.id_col = id_col
        self.text_col = text_col
        self.where = where

    def extract(self) -> Iterator[dict]:
        conn = sqlite3.connect(self.db_path)
        sql = f"SELECT {self.id_col}, {self.text_col} FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        for row in conn.execute(sql):
            yield {"id": str(row[0]), "text": row[1]}
        conn.close()

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        sql = f"SELECT COUNT(*) FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        result = conn.execute(sql).fetchone()[0]
        conn.close()
        return result

    def save_jsonl(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for doc in self.extract():
                f.write(json.dumps(doc) + "\n")


class PgVectorExtractor:
    """Extract source text from a PostgreSQL+pgvector table.

    Requires psycopg2. Install with: pip install embedding_tools[migration]
    """

    def __init__(
        self,
        dbname: str,
        table: str,
        id_col: str = "id",
        text_col: str = "content",
        host: str = "localhost",
        port: int = 5432,
        user: Optional[str] = None,
        where: Optional[str] = None,
    ):
        self.dbname = dbname
        self.table = table
        self.id_col = id_col
        self.text_col = text_col
        self.host = host
        self.port = port
        self.user = user
        self.where = where

    def _connect(self):
        import psycopg2
        import os

        return psycopg2.connect(
            dbname=self.dbname,
            host=self.host,
            port=self.port,
            user=self.user or os.getenv("USER"),
        )

    def extract(self) -> Iterator[dict]:
        conn = self._connect()
        cur = conn.cursor()
        sql = f"SELECT {self.id_col}, {self.text_col} FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        cur.execute(sql)
        for row in cur:
            yield {"id": str(row[0]), "text": row[1]}
        conn.close()

    def count(self) -> int:
        conn = self._connect()
        cur = conn.cursor()
        sql = f"SELECT COUNT(*) FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        cur.execute(sql)
        result = cur.fetchone()[0]
        conn.close()
        return result

    def save_jsonl(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for doc in self.extract():
                f.write(json.dumps(doc) + "\n")
