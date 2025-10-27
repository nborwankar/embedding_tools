"""Tests for configuration and versioning."""

import pytest
from embedding_tools import compute_config_hash, compute_param_hash


class TestConfigVersioning:
    """Tests for configuration hashing."""

    def test_compute_config_hash(self):
        config = {'model': 'bert', 'dim': 768, 'batch_size': 32}
        hash_val = compute_config_hash(config)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16
        assert all(c in '0123456789abcdef' for c in hash_val)

    def test_hash_deterministic(self):
        config = {'model': 'bert', 'dim': 768}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)

        assert hash1 == hash2

    def test_hash_order_independent(self):
        config1 = {'model': 'bert', 'dim': 768, 'batch_size': 32}
        config2 = {'batch_size': 32, 'dim': 768, 'model': 'bert'}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_hash_sensitive_to_values(self):
        config1 = {'model': 'bert', 'dim': 768}
        config2 = {'model': 'bert', 'dim': 512}
        config3 = {'model': 'gpt', 'dim': 768}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_compute_param_hash(self):
        hash1 = compute_param_hash(model='bert', dim=768, batch_size=32)
        hash2 = compute_param_hash(model='bert', dim=768, batch_size=32)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_param_hash_order_independent(self):
        # kwargs are naturally order-independent in Python 3.7+
        hash1 = compute_param_hash(model='bert', dim=768)
        hash2 = compute_param_hash(dim=768, model='bert')

        assert hash1 == hash2

    def test_nested_config_hash(self):
        config1 = {
            'model': 'bert',
            'training': {'lr': 0.001, 'batch_size': 32},
            'data': {'chunk_size': 512}
        }
        config2 = {
            'model': 'bert',
            'training': {'lr': 0.001, 'batch_size': 32},
            'data': {'chunk_size': 512}
        }
        config3 = {
            'model': 'bert',
            'training': {'lr': 0.01, 'batch_size': 32},  # Different lr
            'data': {'chunk_size': 512}
        }

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)

        assert hash1 == hash2
        assert hash1 != hash3
