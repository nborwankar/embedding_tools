"""MLX array backend implementation for Apple Silicon."""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Tuple
import pickle
import os

from .base import ArrayBackend

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class MLXBackend(ArrayBackend):
    """MLX implementation of ArrayBackend (Apple Silicon GPU acceleration)."""

    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Please install mlx with: pip install mlx")

    def create_array(self, data: Any, dtype: Optional[str] = None) -> mx.array:
        """Create MLX array from data."""
        if dtype is None:
            dtype = mx.float32
        else:
            dtype = getattr(mx, dtype, mx.float32)

        if isinstance(data, np.ndarray):
            return mx.array(data, dtype=dtype)
        elif isinstance(data, mx.array):
            return mx.astype(data, dtype)
        else:
            return mx.array(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> mx.array:
        """Create array of zeros."""
        if dtype is None:
            dtype = mx.float32
        else:
            dtype = getattr(mx, dtype, mx.float32)
        return mx.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> mx.array:
        """Create array of ones."""
        if dtype is None:
            dtype = mx.float32
        else:
            dtype = getattr(mx, dtype, mx.float32)
        return mx.ones(shape, dtype=dtype)

    def random_normal(
        self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0
    ) -> mx.array:
        """Create array with random normal distribution."""
        return mx.random.normal(shape, dtype=mx.float32) * std + mean

    def dot(self, a: mx.array, b: mx.array) -> mx.array:
        """Compute dot product."""
        return mx.matmul(a, b)

    def cosine_similarity(self, a: mx.array, b: mx.array) -> mx.array:
        """Compute cosine similarity."""
        # Ensure arrays are 2D
        if a.ndim == 1:
            a = mx.expand_dims(a, 0)
        if b.ndim == 1:
            b = mx.expand_dims(b, 0)

        # Normalize vectors
        a_norm = a / mx.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / mx.linalg.norm(b, axis=1, keepdims=True)

        # Compute cosine similarity
        return mx.matmul(a_norm, mx.transpose(b_norm))

    def normalize(self, a: mx.array, axis: int = -1) -> mx.array:
        """L2 normalize array."""
        return a / mx.linalg.norm(a, axis=axis, keepdims=True)

    def concatenate(self, arrays: List[mx.array], axis: int = 0) -> mx.array:
        """Concatenate arrays."""
        return mx.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[mx.array], axis: int = 0) -> mx.array:
        """Stack arrays."""
        return mx.stack(arrays, axis=axis)

    def slice_last_dim(self, array: mx.array, dim: int) -> mx.array:
        """Slice array to specific dimension along last axis."""
        if array.ndim == 1:
            return array[:dim]
        elif array.ndim == 2:
            return array[:, :dim]
        else:
            # For higher dimensions, slice the last dimension
            slices = [slice(None)] * (array.ndim - 1) + [slice(None, dim)]
            return array[tuple(slices)]

    def to_numpy(self, array: mx.array) -> np.ndarray:
        """Convert MLX array to NumPy."""
        return np.array(array)

    def from_numpy(self, array: np.ndarray) -> mx.array:
        """Convert NumPy array to MLX format."""
        return mx.array(array)

    def save(self, array: mx.array, filepath: str) -> None:
        """Save array to file."""
        # Convert to numpy for saving since MLX doesn't have native file I/O
        numpy_array = self.to_numpy(array)

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            np.save(filepath, numpy_array)
        elif ext == ".npz":
            np.savez_compressed(filepath, data=numpy_array)
        elif ext == ".pkl":
            # Save as MLX array in pickle
            with open(filepath, "wb") as f:
                pickle.dump(array, f)
        else:
            # Default to .npy
            np.save(filepath, numpy_array)

    def load(self, filepath: str) -> mx.array:
        """Load array from file."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            numpy_array = np.load(filepath)
            return self.from_numpy(numpy_array)
        elif ext == ".npz":
            data = np.load(filepath)
            numpy_array = data["data"]
            return self.from_numpy(numpy_array)
        elif ext == ".pkl":
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, mx.array):
                    return loaded
                else:
                    return self.from_numpy(loaded)
        else:
            # Try .npy first
            try:
                numpy_array = np.load(filepath)
                return self.from_numpy(numpy_array)
            except:
                # Fallback to pickle
                with open(filepath, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, mx.array):
                        return loaded
                    else:
                        return self.from_numpy(loaded)

    def get_memory_usage(self, array: mx.array) -> int:
        """Get memory usage in bytes."""
        # MLX arrays don't have direct nbytes, so convert to numpy temporarily
        return self.to_numpy(array).nbytes

    def get_shape(self, array: mx.array) -> Tuple[int, ...]:
        """Get array shape."""
        return tuple(array.shape)

    def get_dtype(self, array: mx.array) -> str:
        """Get array dtype."""
        return str(array.dtype)

    def top_k_cosine_neighbors(
        self,
        queries: mx.array,
        corpus: mx.array,
        k: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Brute-force exact top-k cosine neighbours (MLX / Apple Silicon GPU)."""
        if queries.ndim == 1:
            queries = mx.expand_dims(queries, 0)
        if corpus.ndim == 1:
            corpus = mx.expand_dims(corpus, 0)
        if k > corpus.shape[0]:
            raise ValueError(f"k={k} exceeds corpus size {corpus.shape[0]}")

        # Normalise corpus once
        corpus_norm = corpus / mx.linalg.norm(corpus, axis=1, keepdims=True)

        n = queries.shape[0]
        if batch_size is None or n <= batch_size:
            return _mlx_topk_chunk(queries, corpus_norm, k)

        idx_chunks: List[mx.array] = []
        cos_chunks: List[mx.array] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sub_idx, sub_cos = _mlx_topk_chunk(queries[start:end], corpus_norm, k)
            idx_chunks.append(sub_idx)
            cos_chunks.append(sub_cos)
        return (
            mx.concatenate(idx_chunks, axis=0),
            mx.concatenate(cos_chunks, axis=0),
        )


def _mlx_topk_chunk(queries: mx.array, corpus_norm: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Compute top-k for one chunk; assumes corpus_norm is L2-normalised."""
    q = queries / mx.linalg.norm(queries, axis=1, keepdims=True)
    cos = mx.matmul(q, mx.transpose(corpus_norm))  # (n, m)

    # MLX argsort is stable and fast enough at this scale; no argpartition path
    sorted_idx = mx.argsort(-cos, axis=-1)[:, :k].astype(mx.int32)
    sorted_cos = mx.take_along_axis(cos, sorted_idx, axis=-1)
    mx.eval(sorted_idx, sorted_cos)
    return sorted_idx, sorted_cos
