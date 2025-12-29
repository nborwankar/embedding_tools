"""JAX backend implementation with XLA JIT compilation.

This backend provides GPU acceleration via Metal (Apple Silicon), CUDA (NVIDIA),
or ROCm (AMD), with automatic JIT compilation for performance.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .base import ArrayBackend


class JAXBackend(ArrayBackend):
    """JAX backend with JIT compilation and multi-device support."""

    def __init__(self, device: Optional[str] = None):
        """Initialize JAX backend.

        Args:
            device: Device to use ('gpu', 'cpu', or None for auto-detect)
                   JAX will use first available GPU/TPU if device='gpu'
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Install with:\n"
                "  macOS: pip install embedding_tools[jax]\n"
                "  Linux: pip install embedding_tools[jax]"
            )

        # Get available devices
        devices = jax.devices()

        if device is None:
            # Auto-detect: prefer GPU/TPU over CPU
            self.device = devices[0]  # JAX puts best device first
        elif device == 'gpu':
            gpu_devices = [d for d in devices if d.platform in ('gpu', 'METAL', 'cuda')]
            if not gpu_devices:
                raise ValueError("No GPU devices available")
            self.device = gpu_devices[0]
        elif device == 'cpu':
            cpu_devices = [d for d in devices if d.platform == 'cpu']
            if not cpu_devices:
                raise ValueError("No CPU devices available")
            self.device = cpu_devices[0]
        else:
            raise ValueError(f"Unknown device: {device}. Use 'gpu', 'cpu', or None")

        # Pre-compile common operations for performance
        self._compile_kernels()

    def _compile_kernels(self):
        """Pre-compile frequently used operations with JIT."""

        @jax.jit
        def _cosine_similarity_kernel(a, b):
            """JIT-compiled cosine similarity."""
            # Ensure arrays are 2D
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if b.ndim == 1:
                b = b.reshape(1, -1)

            a_norm = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
            b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
            return jnp.dot(a_norm, b_norm.T)

        # Store compiled kernels
        self._cosine_sim = _cosine_similarity_kernel

    def _normalize_impl(self, a, axis):
        """Implementation of L2 normalization.

        Not JIT-compiled due to dynamic axis parameter.
        """
        return a / jnp.linalg.norm(a, axis=axis, keepdims=True)

    def create_array(self, data: Any, dtype: Optional[str] = None) -> Any:
        """Create JAX array from data."""
        if dtype is None:
            dtype = jnp.float32
        else:
            # Handle string dtype specifications
            dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

        # Convert to JAX array and place on device
        arr = jnp.array(data, dtype=dtype)
        return jax.device_put(arr, self.device)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Any:
        """Create zero-filled array."""
        if dtype is None:
            dtype = jnp.float32
        else:
            dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

        arr = jnp.zeros(shape, dtype=dtype)
        return jax.device_put(arr, self.device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Any:
        """Create one-filled array."""
        if dtype is None:
            dtype = jnp.float32
        else:
            dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

        arr = jnp.ones(shape, dtype=dtype)
        return jax.device_put(arr, self.device)

    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0
    ) -> Any:
        """Create random normal array."""
        # JAX requires explicit PRNG key
        key = jax.random.PRNGKey(0)
        arr = mean + std * jax.random.normal(key, shape, dtype=jnp.float32)
        return jax.device_put(arr, self.device)

    def dot(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        return jnp.dot(a, b)

    def cosine_similarity(self, a: Any, b: Any) -> Any:
        """Compute cosine similarity (uses JIT-compiled kernel)."""
        return self._cosine_sim(a, b)

    def normalize(self, a: Any, axis: int = -1) -> Any:
        """L2 normalization."""
        return self._normalize_impl(a, axis)

    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        """Concatenate arrays."""
        return jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[Any], axis: int = 0) -> Any:
        """Stack arrays."""
        return jnp.stack(arrays, axis=axis)

    def slice_last_dim(self, array: Any, dim: int) -> Any:
        """Slice array to specific dimension on last axis."""
        return array[..., :dim]

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert JAX array to NumPy."""
        return np.array(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert NumPy array to JAX."""
        return self.create_array(array)

    def save(self, array: Any, filepath: str) -> None:
        """Save array to file (converts to NumPy)."""
        # JAX doesn't have native format, use NumPy
        np_array = self.to_numpy(array)
        np.save(filepath, np_array)

    def load(self, filepath: str) -> Any:
        """Load array from file."""
        np_array = np.load(filepath)
        return self.create_array(np_array)

    def get_memory_usage(self, array: Any) -> int:
        """Get memory usage in bytes."""
        return array.nbytes

    def get_shape(self, array: Any) -> Tuple[int, ...]:
        """Get array shape."""
        return array.shape

    def get_dtype(self, array: Any) -> str:
        """Get array dtype as string."""
        dtype_str = str(array.dtype)
        # Clean up JAX-specific prefixes
        return dtype_str.replace('jax.numpy.', '').replace('jnp.', '')
