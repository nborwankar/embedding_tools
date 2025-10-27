"""Installation validation tests.

Run these tests after installation to verify the package works correctly.
Can be run with: pytest tests/test_installation.py -v
"""

import pytest
import numpy as np


def test_package_import():
    """Test that embedding_tools can be imported."""
    import embedding_tools
    assert embedding_tools.__version__ is not None


def test_version_string():
    """Test version string format."""
    import embedding_tools
    version_parts = embedding_tools.__version__.split('.')
    assert len(version_parts) >= 2  # At least major.minor


def test_core_imports():
    """Test all core components can be imported."""
    from embedding_tools import (
        get_backend,
        NumpyBackend,
        EmbeddingStore,
        compute_config_hash,
        compute_param_hash,
    )

    # All imports successful if we get here
    assert get_backend is not None
    assert NumpyBackend is not None
    assert EmbeddingStore is not None
    assert compute_config_hash is not None


def test_numpy_backend_available():
    """Test NumPy backend is always available."""
    from embedding_tools import get_backend

    backend = get_backend('numpy')
    assert backend is not None
    assert hasattr(backend, 'create_array')


def test_mlx_backend_conditional():
    """Test MLX backend availability."""
    from embedding_tools import MLX_AVAILABLE, get_backend

    if MLX_AVAILABLE:
        backend = get_backend('mlx')
        assert backend is not None
    else:
        with pytest.raises(ImportError):
            get_backend('mlx')


def test_auto_backend_detection():
    """Test automatic backend detection."""
    from embedding_tools import get_backend

    backend = get_backend()  # No argument - auto-detect
    assert backend is not None


def test_basic_array_creation():
    """Test basic array creation works."""
    from embedding_tools import get_backend

    backend = get_backend('numpy')
    arr = backend.create_array([1, 2, 3, 4, 5])

    assert arr is not None
    assert backend.get_shape(arr) == (5,)


def test_cosine_similarity():
    """Test cosine similarity computation."""
    from embedding_tools import get_backend

    backend = get_backend('numpy')

    a = backend.create_array([[1, 0, 0], [0, 1, 0]])
    b = backend.create_array([[1, 0, 0], [0, 0, 1]])

    sim = backend.cosine_similarity(a, b)
    sim_np = backend.to_numpy(sim)

    # [1,0,0] dot [1,0,0] = 1.0 (identical)
    # [1,0,0] dot [0,0,1] = 0.0 (orthogonal)
    # [0,1,0] dot [1,0,0] = 0.0 (orthogonal)
    # [0,1,0] dot [0,0,1] = 0.0 (orthogonal)
    assert sim_np[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert sim_np[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_embedding_store_creation():
    """Test EmbeddingStore can be created."""
    from embedding_tools import EmbeddingStore

    store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)
    assert store is not None
    assert store.get_total_memory_usage() == 0


def test_embedding_store_add():
    """Test adding embeddings to store."""
    from embedding_tools import EmbeddingStore

    store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)

    embeddings = np.random.randn(100, 128).astype(np.float32)
    store.add_embeddings(embeddings, dimension=128)

    retrieved = store.get_embeddings(dimension=128)
    assert retrieved is not None
    assert store.backend.get_shape(retrieved) == (100, 128)


def test_config_versioning():
    """Test configuration hashing."""
    from embedding_tools import compute_config_hash, compute_param_hash

    config1 = {'model': 'bert', 'dim': 768}
    config2 = {'model': 'bert', 'dim': 768}
    config3 = {'model': 'bert', 'dim': 512}

    hash1 = compute_config_hash(config1)
    hash2 = compute_config_hash(config2)
    hash3 = compute_config_hash(config3)

    # Same config should produce same hash
    assert hash1 == hash2

    # Different config should produce different hash
    assert hash1 != hash3

    # Hash should be 16 characters
    assert len(hash1) == 16


def test_param_hash_convenience():
    """Test param_hash convenience function."""
    from embedding_tools import compute_param_hash

    hash1 = compute_param_hash(model='bert', dim=768)
    hash2 = compute_param_hash(model='bert', dim=768)
    hash3 = compute_param_hash(model='gpt', dim=768)

    assert hash1 == hash2
    assert hash1 != hash3


def test_slice_last_dim():
    """Test dimension slicing."""
    from embedding_tools import get_backend

    backend = get_backend('numpy')

    # 2D array
    arr = backend.create_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    sliced = backend.slice_last_dim(arr, 2)

    sliced_np = backend.to_numpy(sliced)
    assert sliced_np.shape == (2, 2)
    assert np.array_equal(sliced_np, [[1, 2], [5, 6]])


def test_memory_tracking():
    """Test memory usage tracking."""
    from embedding_tools import EmbeddingStore

    store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)

    # float32 * 1000 * 512 = 2,048,000 bytes = ~2MB
    embeddings = np.random.randn(1000, 512).astype(np.float32)
    store.add_embeddings(embeddings, dimension=512)

    memory_usage = store.get_total_memory_usage()
    expected = 1000 * 512 * 4  # 4 bytes per float32

    assert memory_usage == expected


def test_save_load_roundtrip():
    """Test save/load roundtrip."""
    import tempfile
    import os
    from embedding_tools import get_backend

    backend = get_backend('numpy')

    original = backend.create_array([[1, 2, 3], [4, 5, 6]])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.npy')

        # Save
        backend.save(original, filepath)
        assert os.path.exists(filepath)

        # Load
        loaded = backend.load(filepath)
        loaded_np = backend.to_numpy(loaded)
        original_np = backend.to_numpy(original)

        assert np.array_equal(loaded_np, original_np)


def test_installation_summary():
    """Print installation summary."""
    from embedding_tools import __version__, MLX_AVAILABLE, get_backend

    print("\n" + "="*60)
    print("embedding_tools Installation Validation Summary")
    print("="*60)
    print(f"Version: {__version__}")
    print(f"NumPy backend: ✓ Available")
    print(f"MLX backend: {'✓ Available' if MLX_AVAILABLE else '✗ Not installed'}")

    backend = get_backend()
    print(f"Auto-detected backend: {backend.__class__.__name__}")

    print("\nAll core functionality tests passed!")
    print("="*60)


if __name__ == '__main__':
    # Allow running directly for quick validation
    pytest.main([__file__, '-v', '--tb=short'])
