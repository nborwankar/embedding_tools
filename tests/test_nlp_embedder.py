"""Tests for Embedder model registry and configuration."""

import pytest


class TestRegistry:
    """Test model registry without loading any models."""

    def test_registry_has_nomic(self):
        from embedding_tools.nlp import Embedder

        assert "nomic-v1.5" in Embedder.REGISTRY

    def test_registry_has_jina_code(self):
        from embedding_tools.nlp import Embedder

        assert "jina-code-v2" in Embedder.REGISTRY

    def test_registry_has_legacy_minilm(self):
        from embedding_tools.nlp import Embedder

        assert "minilm-l6" in Embedder.REGISTRY
        assert Embedder.REGISTRY["minilm-l6"].get("deprecated") is True

    def test_registry_has_legacy_mpnet(self):
        from embedding_tools.nlp import Embedder

        assert "mpnet" in Embedder.REGISTRY
        assert Embedder.REGISTRY["mpnet"].get("deprecated") is True

    def test_registry_has_legacy_unixcoder(self):
        from embedding_tools.nlp import Embedder

        assert "unixcoder" in Embedder.REGISTRY
        assert Embedder.REGISTRY["unixcoder"].get("deprecated") is True

    def test_all_entries_have_required_fields(self):
        from embedding_tools.nlp import Embedder

        required = {"hf_name", "dim", "matryoshka", "max_context", "training_data"}
        for key, entry in Embedder.REGISTRY.items():
            missing = required - set(entry.keys())
            assert not missing, f"Registry entry '{key}' missing fields: {missing}"

    def test_default_model_is_nomic(self):
        from embedding_tools.nlp import Embedder

        e = Embedder()
        assert e.model_key == "nomic-v1.5"
        assert e.dim == 768

    def test_matryoshka_truncation_config(self):
        from embedding_tools.nlp import Embedder

        e = Embedder("nomic-v1.5", truncate_dim=256)
        assert e.dim == 256
        assert e.full_dim == 768

    def test_matryoshka_on_non_matryoshka_raises(self):
        from embedding_tools.nlp import Embedder

        with pytest.raises(ValueError, match="does not support Matryoshka"):
            Embedder("minilm-l6", truncate_dim=128)

    def test_invalid_model_key_raises(self):
        from embedding_tools.nlp import Embedder

        with pytest.raises(KeyError):
            Embedder("nonexistent-model")
