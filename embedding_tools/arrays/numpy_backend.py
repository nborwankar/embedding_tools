"""NumPy array backend implementation."""

import numpy as np
from typing import Any, List, Optional, Tuple
import pickle
import os

from .base import ArrayBackend


class NumpyBackend(ArrayBackend):
    """NumPy implementation of ArrayBackend."""

    def create_array(self, data: Any, dtype: Optional[str] = None) -> np.ndarray:
        """Create NumPy array from data."""
        if dtype is None:
            dtype = np.float32
        return np.array(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> np.ndarray:
        """Create array of zeros."""
        if dtype is None:
            dtype = np.float32
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> np.ndarray:
        """Create array of ones."""
        if dtype is None:
            dtype = np.float32
        return np.ones(shape, dtype=dtype)

    def random_normal(
        self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0
    ) -> np.ndarray:
        """Create array with random normal distribution."""
        return np.random.normal(mean, std, shape).astype(np.float32)

    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute dot product."""
        return np.dot(a, b)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        # Ensure arrays are 2D
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

        # Compute cosine similarity
        return np.dot(a_norm, b_norm.T)

    def normalize(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        """L2 normalize array."""
        return a / np.linalg.norm(a, axis=axis, keepdims=True)

    def concatenate(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenate arrays."""
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        """Stack arrays."""
        return np.stack(arrays, axis=axis)

    def slice_last_dim(self, array: np.ndarray, dim: int) -> np.ndarray:
        """Slice array to specific dimension along last axis."""
        if array.ndim == 1:
            return array[:dim]
        elif array.ndim == 2:
            return array[:, :dim]
        else:
            # For higher dimensions, slice the last dimension
            return array[..., :dim]

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        """Convert array to NumPy (already NumPy)."""
        return array

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        """Convert NumPy array to backend format (already NumPy)."""
        return array

    def save(self, array: np.ndarray, filepath: str) -> None:
        """Save array to file."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            np.save(filepath, array)
        elif ext == ".npz":
            np.savez_compressed(filepath, data=array)
        elif ext == ".pkl":
            with open(filepath, "wb") as f:
                pickle.dump(array, f)
        else:
            # Default to .npy
            np.save(filepath, array)

    def load(self, filepath: str) -> np.ndarray:
        """Load array from file."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            return np.load(filepath)
        elif ext == ".npz":
            data = np.load(filepath)
            return data["data"]
        elif ext == ".pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            # Try .npy first
            try:
                return np.load(filepath)
            except:
                # Fallback to pickle
                with open(filepath, "rb") as f:
                    return pickle.load(f)

    def get_memory_usage(self, array: np.ndarray) -> int:
        """Get memory usage in bytes."""
        return array.nbytes

    def get_shape(self, array: np.ndarray) -> Tuple[int, ...]:
        """Get array shape."""
        return array.shape

    def get_dtype(self, array: np.ndarray) -> str:
        """Get array dtype."""
        return str(array.dtype)

    def top_k_cosine_neighbors(
        self,
        queries: np.ndarray,
        corpus: np.ndarray,
        k: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Brute-force exact top-k cosine neighbours (NumPy)."""
        queries = np.asarray(queries)
        corpus = np.asarray(corpus)
        if queries.ndim == 1:
            queries = queries[None, :]
        if corpus.ndim == 1:
            corpus = corpus[None, :]
        if k > corpus.shape[0]:
            raise ValueError(f"k={k} exceeds corpus size {corpus.shape[0]}")

        # Normalise corpus once
        corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

        n = queries.shape[0]
        if batch_size is None or n <= batch_size:
            return _numpy_topk_chunk(queries, corpus_norm, k)

        idx_chunks: List[np.ndarray] = []
        cos_chunks: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sub_idx, sub_cos = _numpy_topk_chunk(queries[start:end], corpus_norm, k)
            idx_chunks.append(sub_idx)
            cos_chunks.append(sub_cos)
        return (
            np.concatenate(idx_chunks, axis=0),
            np.concatenate(cos_chunks, axis=0),
        )


def _numpy_topk_chunk(
    queries: np.ndarray, corpus_norm: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-k for one chunk; assumes corpus_norm is L2-normalised."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    cos = q @ corpus_norm.T  # (n, m)

    if k == cos.shape[1]:
        sorted_idx = np.argsort(-cos, axis=-1)
    else:
        # argpartition gives unordered top-k; reorder by value
        part = np.argpartition(-cos, kth=k - 1, axis=-1)[:, :k]
        within = np.argsort(-np.take_along_axis(cos, part, axis=-1), axis=-1)
        sorted_idx = np.take_along_axis(part, within, axis=-1)

    sorted_cos = np.take_along_axis(cos, sorted_idx, axis=-1)
    return sorted_idx.astype(np.int64), sorted_cos.astype(queries.dtype)
