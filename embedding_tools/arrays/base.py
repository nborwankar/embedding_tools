"""Abstract base class for array backends.

This module provides a backend-agnostic interface for array operations,
allowing seamless switching between NumPy, MLX, and PyTorch implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import numpy as np


class ArrayBackend(ABC):
    """Abstract base class for different array backends (NumPy, MLX, PyTorch)."""

    @abstractmethod
    def create_array(self, data: Any, dtype: Optional[str] = None) -> Any:
        """Create array from data.

        Args:
            data: Input data (list, numpy array, etc.)
            dtype: Data type (optional, defaults to float32)

        Returns:
            Array in backend format
        """
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Any:
        """Create array of zeros.

        Args:
            shape: Array shape
            dtype: Data type (optional, defaults to float32)

        Returns:
            Zero-filled array
        """
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Any:
        """Create array of ones.

        Args:
            shape: Array shape
            dtype: Data type (optional, defaults to float32)

        Returns:
            One-filled array
        """
        pass

    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> Any:
        """Create array with random normal distribution.

        Args:
            shape: Array shape
            mean: Distribution mean (default: 0.0)
            std: Distribution standard deviation (default: 1.0)

        Returns:
            Random array from normal distribution
        """
        pass

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """Compute dot product.

        Args:
            a: First array
            b: Second array

        Returns:
            Dot product result
        """
        pass

    @abstractmethod
    def cosine_similarity(self, a: Any, b: Any) -> Any:
        """Compute cosine similarity between arrays.

        Args:
            a: First array (can be 1D or 2D)
            b: Second array (can be 1D or 2D)

        Returns:
            Cosine similarity matrix
        """
        pass

    @abstractmethod
    def normalize(self, a: Any, axis: int = -1) -> Any:
        """L2 normalize array.

        Args:
            a: Input array
            axis: Axis along which to normalize (default: -1)

        Returns:
            Normalized array
        """
        pass

    @abstractmethod
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        """Concatenate arrays along axis.

        Args:
            arrays: List of arrays to concatenate
            axis: Concatenation axis (default: 0)

        Returns:
            Concatenated array
        """
        pass

    @abstractmethod
    def stack(self, arrays: List[Any], axis: int = 0) -> Any:
        """Stack arrays along new axis.

        Args:
            arrays: List of arrays to stack
            axis: New axis position (default: 0)

        Returns:
            Stacked array
        """
        pass

    @abstractmethod
    def slice_last_dim(self, array: Any, dim: int) -> Any:
        """Slice array to specific dimension along last axis.

        Useful for truncating embeddings to lower dimensions.

        Args:
            array: Input array (1D, 2D, or higher)
            dim: Target dimension size

        Returns:
            Sliced array with shape [..., dim]
        """
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert array to NumPy format.

        Args:
            array: Input array in backend format

        Returns:
            NumPy array
        """
        pass

    @abstractmethod
    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert NumPy array to backend format.

        Args:
            array: NumPy array

        Returns:
            Array in backend format
        """
        pass

    @abstractmethod
    def save(self, array: Any, filepath: str) -> None:
        """Save array to file.

        Args:
            filepath: Path to save file (.npy, .npz, .pkl supported)
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load array from file.

        Args:
            filepath: Path to load from

        Returns:
            Loaded array in backend format
        """
        pass

    @abstractmethod
    def get_memory_usage(self, array: Any) -> int:
        """Get memory usage in bytes.

        Args:
            array: Input array

        Returns:
            Memory usage in bytes
        """
        pass

    @abstractmethod
    def get_shape(self, array: Any) -> Tuple[int, ...]:
        """Get array shape.

        Args:
            array: Input array

        Returns:
            Shape tuple
        """
        pass

    @abstractmethod
    def get_dtype(self, array: Any) -> str:
        """Get array dtype as string.

        Args:
            array: Input array

        Returns:
            Data type string
        """
        pass


def get_backend(backend_name: Optional[str] = None, device: Optional[str] = None) -> ArrayBackend:
    """Get array backend by name.

    Args:
        backend_name: Backend name ('numpy', 'mlx', 'torch', 'jax').
                     If None, auto-detect best available backend.
        device: Device for PyTorch/JAX backend ('cuda', 'mps', 'cpu', 'gpu').
               Only used when backend_name='torch' or 'jax'. Auto-detects if None.

    Returns:
        ArrayBackend instance

    Raises:
        ValueError: If backend_name is unknown
        ImportError: If requested backend is not available
    """
    # Auto-detect if not specified
    if backend_name is None:
        try:
            import mlx.core
            backend_name = 'mlx'
        except ImportError:
            try:
                import jax
                backend_name = 'jax'
            except ImportError:
                try:
                    import torch
                    backend_name = 'torch'
                except ImportError:
                    backend_name = 'numpy'

    backend_name = backend_name.lower()

    if backend_name == "numpy":
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    elif backend_name == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(device=device)
    elif backend_name == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend(device=device)
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Supported: 'numpy', 'mlx', 'torch', 'jax'"
        )
