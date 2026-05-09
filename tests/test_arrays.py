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
        assert backend.get_dtype(arr) == "float32"

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


class TestTopKCosineNeighbors:
    """Cross-backend tests for top_k_cosine_neighbors.

    Numpy is the ground truth; MLX/Torch/JAX must agree to fp tolerance.
    """

    @pytest.fixture
    def fixture(self):
        rng = np.random.default_rng(42)
        # Small enough to brute-force verify by hand
        corpus = rng.standard_normal((50, 16)).astype(np.float32)
        queries = rng.standard_normal((7, 16)).astype(np.float32)
        return queries, corpus

    def _ground_truth(self, queries, corpus, k):
        # Brute-force: normalise both, dot, argsort descending, take top-k
        c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        cos = q @ c.T
        idx = np.argsort(-cos, axis=-1)[:, :k]
        sims = np.take_along_axis(cos, idx, axis=-1)
        return idx.astype(np.int64), sims.astype(queries.dtype)

    def test_numpy_matches_ground_truth(self, fixture):
        from embedding_tools import NumpyBackend

        queries, corpus = fixture
        backend = NumpyBackend()
        gt_idx, gt_sim = self._ground_truth(queries, corpus, k=5)
        idx, sim = backend.top_k_cosine_neighbors(queries, corpus, k=5)
        np.testing.assert_array_equal(idx, gt_idx)
        np.testing.assert_allclose(sim, gt_sim, atol=1e-6)

    def test_numpy_batched_matches_unbatched(self, fixture):
        from embedding_tools import NumpyBackend

        queries, corpus = fixture
        backend = NumpyBackend()
        idx_unbatched, sim_unbatched = backend.top_k_cosine_neighbors(queries, corpus, k=5)
        idx_batched, sim_batched = backend.top_k_cosine_neighbors(
            queries, corpus, k=5, batch_size=3
        )
        np.testing.assert_array_equal(idx_batched, idx_unbatched)
        np.testing.assert_allclose(sim_batched, sim_unbatched, atol=1e-6)

    def test_numpy_k_equals_corpus_size(self, fixture):
        from embedding_tools import NumpyBackend

        queries, corpus = fixture
        backend = NumpyBackend()
        idx, sim = backend.top_k_cosine_neighbors(queries, corpus, k=corpus.shape[0])
        assert idx.shape == (queries.shape[0], corpus.shape[0])
        # Each row of idx must be a permutation of [0..M-1]
        for row in idx:
            assert sorted(row.tolist()) == list(range(corpus.shape[0]))
        # Cosines must be sorted descending
        for row in sim:
            assert all(row[i] >= row[i + 1] - 1e-6 for i in range(len(row) - 1))

    def test_k_too_large_raises(self, fixture):
        from embedding_tools import NumpyBackend

        queries, corpus = fixture
        backend = NumpyBackend()
        with pytest.raises(ValueError, match="exceeds corpus size"):
            backend.top_k_cosine_neighbors(queries, corpus, k=corpus.shape[0] + 1)

    def test_1d_input_treated_as_single_row(self):
        from embedding_tools import NumpyBackend

        backend = NumpyBackend()
        rng = np.random.default_rng(0)
        corpus = rng.standard_normal((10, 4)).astype(np.float32)
        single_query = rng.standard_normal(4).astype(np.float32)
        idx, sim = backend.top_k_cosine_neighbors(single_query, corpus, k=3)
        assert idx.shape == (1, 3)
        assert sim.shape == (1, 3)

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
    def test_mlx_matches_numpy(self, fixture):
        from embedding_tools import MLXBackend, NumpyBackend

        queries, corpus = fixture
        np_backend = NumpyBackend()
        try:
            mlx_backend = MLXBackend()
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"MLX runtime not usable: {e}")
        np_idx, np_sim = np_backend.top_k_cosine_neighbors(queries, corpus, k=5)
        mx_idx, mx_sim = mlx_backend.top_k_cosine_neighbors(
            mlx_backend.from_numpy(queries),
            mlx_backend.from_numpy(corpus),
            k=5,
        )
        mx_idx_np = mlx_backend.to_numpy(mx_idx)
        mx_sim_np = mlx_backend.to_numpy(mx_sim)
        np.testing.assert_array_equal(mx_idx_np, np_idx)
        np.testing.assert_allclose(mx_sim_np, np_sim, atol=1e-3)


class TestBackendSelection:
    """Test backend selection logic."""

    def test_explicit_numpy(self):
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_auto_detection(self):
        backend = get_backend()  # Auto-detect
        assert backend is not None

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")

    def test_mlx_when_unavailable(self):
        if not MLX_AVAILABLE:
            with pytest.raises(ImportError):
                get_backend("mlx")
