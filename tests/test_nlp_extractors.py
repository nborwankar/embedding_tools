"""Tests for source text extractors."""

import json
import sqlite3

import pytest

from embedding_tools.nlp.extractors import SQLiteExtractor, JSONLExtractor


class TestJSONLExtractor:
    def test_extracts_from_jsonl(self, tmp_path):
        f = tmp_path / "source.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps({"id": "1", "text": "hello", "other": "ignored"}),
                    json.dumps({"id": "2", "text": "world"}),
                ]
            )
        )
        ext = JSONLExtractor(str(f), id_col="id", text_col="text")
        docs = list(ext.extract())
        assert len(docs) == 2
        assert docs[0] == {"id": "1", "text": "hello"}
        assert docs[1] == {"id": "2", "text": "world"}

    def test_counts(self, tmp_path):
        f = tmp_path / "source.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps({"id": "1", "text": "hello"}),
                    json.dumps({"id": "2", "text": "world"}),
                ]
            )
        )
        ext = JSONLExtractor(str(f), id_col="id", text_col="text")
        assert ext.count() == 2

    def test_save_jsonl(self, tmp_path):
        f = tmp_path / "source.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps({"id": "1", "text": "hello"}),
                    json.dumps({"id": "2", "text": "world"}),
                ]
            )
        )
        ext = JSONLExtractor(str(f), id_col="id", text_col="text")
        out = tmp_path / "output.jsonl"
        ext.save_jsonl(str(out))
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": "1", "text": "hello"}

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "source.jsonl"
        f.write_text(
            json.dumps({"id": "1", "text": "hello"})
            + "\n\n"
            + json.dumps({"id": "2", "text": "world"})
            + "\n"
        )
        ext = JSONLExtractor(str(f), id_col="id", text_col="text")
        assert ext.count() == 2


class TestSQLiteExtractor:
    def _make_db(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT, embedding BLOB)")
        conn.execute("INSERT INTO docs VALUES (1, 'hello world', X'0000')")
        conn.execute("INSERT INTO docs VALUES (2, 'foo bar', X'0000')")
        conn.commit()
        conn.close()
        return str(db)

    def test_extracts_from_sqlite(self, tmp_path):
        db = self._make_db(tmp_path)
        ext = SQLiteExtractor(db, table="docs", id_col="id", text_col="content")
        docs = list(ext.extract())
        assert len(docs) == 2
        assert docs[0] == {"id": "1", "text": "hello world"}

    def test_counts(self, tmp_path):
        db = self._make_db(tmp_path)
        ext = SQLiteExtractor(db, table="docs", id_col="id", text_col="content")
        assert ext.count() == 2

    def test_where_clause(self, tmp_path):
        db = self._make_db(tmp_path)
        ext = SQLiteExtractor(db, table="docs", id_col="id", text_col="content", where="id = 1")
        docs = list(ext.extract())
        assert len(docs) == 1
        assert docs[0]["text"] == "hello world"

    def test_save_jsonl(self, tmp_path):
        db = self._make_db(tmp_path)
        ext = SQLiteExtractor(db, table="docs", id_col="id", text_col="content")
        out = tmp_path / "output.jsonl"
        ext.save_jsonl(str(out))
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
