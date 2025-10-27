"""Tests for memory management."""

import pytest
import numpy as np
from embedding_tools import EmbeddingStore


class TestEmbeddingStore:
    """Tests for EmbeddingStore."""

    @pytest.fixture
    def store(self):
        return EmbeddingStore(backend='numpy', max_memory_gb=1.0)

    def test_initialization(self, store):
        assert store is not None
        assert store.get_total_memory_usage() == 0
        assert len(store.get_available_dimensions()) == 0

    def test_add_embeddings(self, store):
        embeddings = np.random.randn(100, 128).astype(np.float32)
        store.add_embeddings(embeddings, dimension=128)

        assert 128 in store.get_available_dimensions()
        retrieved = store.get_embeddings(dimension=128)
        assert store.backend.get_shape(retrieved) == (100, 128)

    def test_memory_limit(self, store):
        # Try to add more than 1GB
        # float32 = 4 bytes, so 1GB = 256M floats = 256M / 512 = 500K vectors of 512D
        large_embeddings = np.random.randn(600000, 512).astype(np.float32)

        with pytest.raises(MemoryError, match="exceed memory limit"):
            store.add_embeddings(large_embeddings, dimension=512)

    def test_multiple_dimensions(self, store):
        emb_128 = np.random.randn(100, 128).astype(np.float32)
        emb_256 = np.random.randn(100, 256).astype(np.float32)

        store.add_embeddings(emb_128, dimension=128)
        store.add_embeddings(emb_256, dimension=256)

        assert set(store.get_available_dimensions()) == {128, 256}

    def test_metadata_storage(self, store):
        embeddings = np.random.randn(10, 64).astype(np.float32)
        text_ids = [f"doc_{i}" for i in range(10)]
        labels = [f"label_{i%3}" for i in range(10)]

        store.add_embeddings(
            embeddings,
            dimension=64,
            text_ids=text_ids,
            labels=labels
        )

        assert store.get_text_ids() == text_ids
        assert store.get_labels() == labels

    def test_slice_to_dimension(self, store):
        # Add 256D embeddings
        emb_256 = np.random.randn(100, 256).astype(np.float32)
        store.add_embeddings(emb_256, dimension=256)

        # Slice to 128D
        emb_128 = store.slice_to_dimension(source_dim=256, target_dim=128)

        assert emb_128 is not None
        assert store.backend.get_shape(emb_128) == (100, 128)
        assert 128 in store.get_available_dimensions()

        # Verify slicing correctness
        original_128 = store.backend.to_numpy(store.get_embeddings(256))[:, :128]
        sliced_128 = store.backend.to_numpy(emb_128)
        np.testing.assert_array_equal(sliced_128, original_128)

    def test_slice_invalid_dimensions(self, store):
        emb_128 = np.random.randn(100, 128).astype(np.float32)
        store.add_embeddings(emb_128, dimension=128)

        # Cannot slice to larger dimension
        with pytest.raises(ValueError, match="Target dimension.*> source dimension"):
            store.slice_to_dimension(source_dim=128, target_dim=256)

    def test_similarity_search(self, store):
        # Create simple test embeddings
        embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ], dtype=np.float32)

        store.add_embeddings(embeddings, dimension=3)

        # Query with vector close to first embedding
        query = np.array([1, 0, 0], dtype=np.float32)
        similarities, indices = store.compute_similarity(query, dimension=3, top_k=2)

        indices_np = store.backend.to_numpy(indices)
        assert indices_np[0] == 0  # Most similar should be index 0

    def test_memory_info(self, store):
        emb_128 = np.random.randn(100, 128).astype(np.float32)
        emb_256 = np.random.randn(100, 256).astype(np.float32)

        store.add_embeddings(emb_128, dimension=128)
        store.add_embeddings(emb_256, dimension=256)

        info = store.get_memory_info()

        assert 'total_bytes' in info
        assert 'total_gb' in info
        assert 'dimensions' in info
        assert 128 in info['dimensions']
        assert 256 in info['dimensions']

        # Check memory calculation
        expected_128 = 100 * 128 * 4  # float32
        expected_256 = 100 * 256 * 4
        assert info['dimensions'][128]['memory_bytes'] == expected_128
        assert info['dimensions'][256]['memory_bytes'] == expected_256

    def test_save_load_roundtrip(self, store, tmp_path):
        embeddings = np.random.randn(50, 64).astype(np.float32)
        store.add_embeddings(embeddings, dimension=64)

        # Save
        store.save_to_disk(tmp_path)

        # Create new store and load
        new_store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)
        new_store.load_from_disk(tmp_path)

        # Verify
        assert 64 in new_store.get_available_dimensions()
        loaded = new_store.get_embeddings(dimension=64)
        loaded_np = new_store.backend.to_numpy(loaded)
        np.testing.assert_array_almost_equal(loaded_np, embeddings)
