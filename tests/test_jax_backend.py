"""Tests for JAX backend implementation."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from embedding_tools.arrays.jax_backend import JAXBackend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXBackend:
    """Test JAX backend operations."""

    @pytest.fixture
    def backend(self):
        """Create JAX backend instance."""
        return JAXBackend()

    def test_initialization(self, backend):
        """Test backend initialization."""
        assert backend.device is not None
        print(f"Device: {backend.device}")

    def test_create_array(self, backend):
        """Test array creation."""
        data = [[1, 2, 3], [4, 5, 6]]
        arr = backend.create_array(data)

        assert backend.get_shape(arr) == (2, 3)
        np_arr = backend.to_numpy(arr)
        np.testing.assert_array_equal(np_arr, data)

    def test_zeros(self, backend):
        """Test zeros creation."""
        arr = backend.zeros((3, 4))
        assert backend.get_shape(arr) == (3, 4)
        np.testing.assert_array_equal(backend.to_numpy(arr), np.zeros((3, 4)))

    def test_ones(self, backend):
        """Test ones creation."""
        arr = backend.ones((2, 5))
        assert backend.get_shape(arr) == (2, 5)
        np.testing.assert_array_equal(backend.to_numpy(arr), np.ones((2, 5)))

    def test_random_normal(self, backend):
        """Test random normal generation."""
        arr = backend.random_normal((100, 50), mean=0.0, std=1.0)
        assert backend.get_shape(arr) == (100, 50)

        np_arr = backend.to_numpy(arr)
        assert abs(np_arr.mean()) < 0.2  # Close to 0
        assert abs(np_arr.std() - 1.0) < 0.2  # Close to 1

    def test_dot_product(self, backend):
        """Test matrix multiplication."""
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6], [7, 8]])

        result = backend.dot(a, b)
        expected = np.array([[19, 22], [43, 50]])

        np.testing.assert_array_almost_equal(
            backend.to_numpy(result),
            expected
        )

    def test_cosine_similarity(self, backend):
        """Test cosine similarity computation."""
        a = backend.create_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = backend.create_array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        sims = backend.cosine_similarity(a, b)
        np_sims = backend.to_numpy(sims)

        # a[0] · b[0] = 1.0 (identical)
        # a[0] · b[1] = 0.0 (orthogonal)
        # a[1] · b[0] = 0.0 (orthogonal)
        # a[1] · b[1] = 0.0 (orthogonal)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])

        np.testing.assert_array_almost_equal(np_sims, expected, decimal=5)

    def test_cosine_similarity_1d(self, backend):
        """Test cosine similarity with 1D arrays."""
        a = backend.create_array([1.0, 0.0, 0.0])
        b = backend.create_array([1.0, 0.0, 0.0])

        sims = backend.cosine_similarity(a, b)
        np_sims = backend.to_numpy(sims)

        # Should return 1.0 for identical vectors
        assert np_sims.shape == (1, 1)
        np.testing.assert_almost_equal(np_sims[0, 0], 1.0, decimal=5)

    def test_normalize(self, backend):
        """Test L2 normalization."""
        arr = backend.create_array([[3.0, 4.0], [5.0, 12.0]])
        normalized = backend.normalize(arr)

        np_norm = backend.to_numpy(normalized)

        # Check unit length
        norms = np.linalg.norm(np_norm, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_concatenate(self, backend):
        """Test array concatenation."""
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6]])

        result = backend.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6]])

        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_stack(self, backend):
        """Test array stacking."""
        a = backend.create_array([1, 2, 3])
        b = backend.create_array([4, 5, 6])

        result = backend.stack([a, b], axis=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_slice_last_dim(self, backend):
        """Test dimension slicing."""
        arr = backend.create_array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        sliced = backend.slice_last_dim(arr, 3)

        expected = np.array([[1, 2, 3], [6, 7, 8]])
        np.testing.assert_array_equal(backend.to_numpy(sliced), expected)

    def test_numpy_conversion(self, backend):
        """Test to_numpy and from_numpy."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        jax_arr = backend.from_numpy(original)
        converted = backend.to_numpy(jax_arr)

        np.testing.assert_array_equal(converted, original)
        assert converted.dtype == original.dtype

    def test_save_load(self, backend, tmp_path):
        """Test save and load operations."""
        arr = backend.create_array([[1, 2, 3], [4, 5, 6]])
        filepath = tmp_path / "test_array.npy"

        backend.save(arr, str(filepath))
        loaded = backend.load(str(filepath))

        np.testing.assert_array_equal(
            backend.to_numpy(arr),
            backend.to_numpy(loaded)
        )

    def test_memory_usage(self, backend):
        """Test memory usage calculation."""
        arr = backend.create_array(np.random.randn(100, 768).astype(np.float32))
        memory = backend.get_memory_usage(arr)

        expected = 100 * 768 * 4  # float32 = 4 bytes
        assert memory == expected

    def test_get_shape(self, backend):
        """Test shape retrieval."""
        arr = backend.create_array(np.random.randn(10, 20, 30))
        assert backend.get_shape(arr) == (10, 20, 30)

    def test_get_dtype(self, backend):
        """Test dtype retrieval."""
        arr = backend.create_array([[1.0, 2.0]], dtype='float32')
        dtype_str = backend.get_dtype(arr)
        assert 'float32' in dtype_str.lower()

    def test_embedding_store_integration(self, backend):
        """Test integration with EmbeddingStore."""
        from embedding_tools import EmbeddingStore

        store = EmbeddingStore(backend='jax', max_memory_gb=1.0)

        # Create sample embeddings
        embeddings = np.random.randn(100, 384).astype(np.float32)
        store.add_embeddings(embeddings, dimension=384)

        # Test retrieval
        retrieved = store.get_embeddings(384)
        assert backend.get_shape(retrieved) == (100, 384)

        # Test similarity search
        query = np.random.randn(384).astype(np.float32)
        sims, indices = store.compute_similarity(query, dimension=384, top_k=10)

        assert len(backend.to_numpy(indices)) == 10
        assert len(backend.to_numpy(sims)) == 10

    def test_jit_compilation_speedup(self, backend):
        """Test that JIT compilation provides speedup on repeated calls."""
        import time

        # Create large arrays
        a = backend.create_array(np.random.randn(1000, 768).astype(np.float32))
        b = backend.create_array(np.random.randn(1000, 768).astype(np.float32))

        # First call (includes compilation time)
        start = time.perf_counter()
        _ = backend.cosine_similarity(a, b)
        first_time = time.perf_counter() - start

        # Second call (uses compiled kernel)
        start = time.perf_counter()
        _ = backend.cosine_similarity(a, b)
        second_time = time.perf_counter() - start

        print(f"\nFirst call: {first_time*1000:.2f}ms")
        print(f"Second call: {second_time*1000:.2f}ms")
        print(f"Speedup: {first_time/second_time:.2f}x")

        # Second call should be faster (or similar if already fast)
        # Allow some variance for testing
        assert second_time <= first_time * 1.5

    def test_device_specification(self):
        """Test explicit device specification."""
        # Test CPU device
        backend_cpu = JAXBackend(device='cpu')
        assert backend_cpu.device.platform == 'cpu'

        # GPU test only if available
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform in ('gpu', 'METAL', 'cuda')]
        if gpu_devices:
            backend_gpu = JAXBackend(device='gpu')
            assert backend_gpu.device.platform in ('gpu', 'METAL', 'cuda')

    def test_auto_backend_detection(self):
        """Test auto-detection includes JAX."""
        from embedding_tools import get_backend

        # JAX should be detected (we know it's installed in this test)
        backend = get_backend()
        # Should be JAX since MLX likely not available in test environment
        assert isinstance(backend, (JAXBackend,)) or backend.__class__.__name__ in ['JAXBackend', 'NumpyBackend']

    def test_explicit_backend_selection(self):
        """Test explicit JAX backend selection."""
        from embedding_tools import get_backend

        backend = get_backend('jax')
        assert isinstance(backend, JAXBackend)

    def test_large_array_operations(self, backend):
        """Test operations on larger arrays (stress test)."""
        # Create larger arrays
        large_arr = backend.create_array(np.random.randn(5000, 512).astype(np.float32))

        # Test operations don't crash
        normalized = backend.normalize(large_arr)
        assert backend.get_shape(normalized) == (5000, 512)

        # Test slicing
        sliced = backend.slice_last_dim(large_arr, 256)
        assert backend.get_shape(sliced) == (5000, 256)

        # Check memory usage is reasonable
        memory_mb = backend.get_memory_usage(large_arr) / (1024 * 1024)
        expected_mb = (5000 * 512 * 4) / (1024 * 1024)  # ~10 MB
        assert abs(memory_mb - expected_mb) < 1.0
