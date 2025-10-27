"""Tests for array backends."""

import pytest
import numpy as np
from embedding_tools import get_backend, NumpyBackend, MLX_AVAILABLE


class TestNumpyBackend:
    """Tests for NumPy backend."""

    @pytest.fixture
    def backend(self):
        return NumpyBackend()

    def test_create_array(self, backend):
        arr = backend.create_array([1, 2, 3, 4, 5])
        assert backend.get_shape(arr) == (5,)
        assert backend.get_dtype(arr) == 'float32'

    def test_zeros(self, backend):
        arr = backend.zeros((3, 4))
        assert backend.get_shape(arr) == (3, 4)
        np.testing.assert_array_equal(backend.to_numpy(arr), np.zeros((3, 4), dtype=np.float32))

    def test_ones(self, backend):
        arr = backend.ones((2, 3))
        np.testing.assert_array_equal(backend.to_numpy(arr), np.ones((2, 3), dtype=np.float32))

    def test_random_normal(self, backend):
        arr = backend.random_normal((100, 50))
        arr_np = backend.to_numpy(arr)
        # Check shape
        assert arr_np.shape == (100, 50)
        # Check approximately normal distribution
        assert abs(arr_np.mean()) < 0.2  # Should be close to 0
        assert abs(arr_np.std() - 1.0) < 0.2  # Should be close to 1

    def test_dot_product(self, backend):
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6], [7, 8]])
        result = backend.dot(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_cosine_similarity_2d(self, backend):
        a = backend.create_array([[1, 0, 0], [0, 1, 0]])
        b = backend.create_array([[1, 0, 0], [0, 0, 1]])
        sim = backend.cosine_similarity(a, b)
        sim_np = backend.to_numpy(sim)

        # Check shape
        assert sim_np.shape == (2, 2)

        # Check values
        assert sim_np[0, 0] == pytest.approx(1.0, abs=1e-6)  # Same vector
        assert sim_np[0, 1] == pytest.approx(0.0, abs=1e-6)  # Orthogonal
        assert sim_np[1, 0] == pytest.approx(0.0, abs=1e-6)  # Orthogonal
        assert sim_np[1, 1] == pytest.approx(0.0, abs=1e-6)  # Orthogonal

    def test_normalize(self, backend):
        a = backend.create_array([[3, 4], [5, 12]])  # 3-4-5 and 5-12-13 triangles
        normalized = backend.normalize(a, axis=1)
        norms = np.linalg.norm(backend.to_numpy(normalized), axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_concatenate(self, backend):
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6], [7, 8]])
        result = backend.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_stack(self, backend):
        a = backend.create_array([1, 2, 3])
        b = backend.create_array([4, 5, 6])
        result = backend.stack([a, b], axis=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_slice_last_dim_1d(self, backend):
        arr = backend.create_array([1, 2, 3, 4, 5])
        sliced = backend.slice_last_dim(arr, 3)
        np.testing.assert_array_equal(backend.to_numpy(sliced), [1, 2, 3])

    def test_slice_last_dim_2d(self, backend):
        arr = backend.create_array([[1, 2, 3, 4], [5, 6, 7, 8]])
        sliced = backend.slice_last_dim(arr, 2)
        np.testing.assert_array_equal(backend.to_numpy(sliced), [[1, 2], [5, 6]])

    def test_memory_usage(self, backend):
        arr = backend.create_array(np.random.randn(100, 256).astype(np.float32))
        memory = backend.get_memory_usage(arr)
        expected = 100 * 256 * 4  # float32 = 4 bytes
        assert memory == expected


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
class TestMLXBackend:
    """Tests for MLX backend (only if available)."""

    @pytest.fixture
    def backend(self):
        from embedding_tools import MLXBackend
        return MLXBackend()

    def test_create_array(self, backend):
        arr = backend.create_array([1, 2, 3, 4, 5])
        assert backend.get_shape(arr) == (5,)

    def test_cosine_similarity(self, backend):
        a = backend.create_array([[1, 0, 0], [0, 1, 0]])
        b = backend.create_array([[1, 0, 0], [0, 0, 1]])
        sim = backend.cosine_similarity(a, b)
        sim_np = backend.to_numpy(sim)

        assert sim_np[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert sim_np[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_numpy_conversion(self, backend):
        arr_mlx = backend.create_array([1, 2, 3])
        arr_np = backend.to_numpy(arr_mlx)
        assert isinstance(arr_np, np.ndarray)
        np.testing.assert_array_equal(arr_np, [1, 2, 3])

        # Round trip
        arr_mlx2 = backend.from_numpy(arr_np)
        arr_np2 = backend.to_numpy(arr_mlx2)
        np.testing.assert_array_equal(arr_np2, [1, 2, 3])


class TestBackendSelection:
    """Test backend selection logic."""

    def test_explicit_numpy(self):
        backend = get_backend('numpy')
        assert isinstance(backend, NumpyBackend)

    def test_auto_detection(self):
        backend = get_backend()  # Auto-detect
        assert backend is not None

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend('invalid')

    def test_mlx_when_unavailable(self):
        if not MLX_AVAILABLE:
            with pytest.raises(ImportError):
                get_backend('mlx')
