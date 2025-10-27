"""PyTorch backend implementation with CUDA/MPS/CPU support.

This backend provides GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon),
with automatic fallback to CPU when GPUs are unavailable.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import ArrayBackend


class TorchBackend(ArrayBackend):
    """PyTorch backend with automatic device selection (CUDA/MPS/CPU)."""

    def __init__(self, device: Optional[str] = None):
        """Initialize PyTorch backend.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        print(f"TorchBackend using device: {self.device}")

    def create_array(self, data: Any, dtype: Optional[str] = None) -> torch.Tensor:
        """Create tensor from data."""
        if dtype is None:
            dtype = 'float32'

        torch_dtype = self._get_torch_dtype(dtype)

        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=torch_dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=self.device, dtype=torch_dtype)
        else:
            return torch.tensor(data, device=self.device, dtype=torch_dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> torch.Tensor:
        """Create tensor of zeros."""
        if dtype is None:
            dtype = 'float32'
        torch_dtype = self._get_torch_dtype(dtype)
        return torch.zeros(shape, device=self.device, dtype=torch_dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> torch.Tensor:
        """Create tensor of ones."""
        if dtype is None:
            dtype = 'float32'
        torch_dtype = self._get_torch_dtype(dtype)
        return torch.ones(shape, device=self.device, dtype=torch_dtype)

    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Create tensor with random normal distribution."""
        return torch.normal(mean, std, size=shape, device=self.device)

    def dot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute dot product."""
        return torch.matmul(a, b)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between tensors.

        Args:
            a: First tensor (can be 1D or 2D)
            b: Second tensor (can be 1D or 2D)

        Returns:
            Cosine similarity matrix
        """
        # Ensure tensors are 2D
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)

        # Normalize
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)

        # Compute similarity
        similarities = torch.mm(a_norm, b_norm.t())

        # Return as 1D if inputs were 1D
        if similarities.shape[0] == 1:
            return similarities.squeeze(0)

        return similarities

    def normalize(self, a: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """L2 normalize tensor."""
        return F.normalize(a, p=2, dim=axis)

    def concatenate(self, arrays: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
        """Concatenate tensors along axis."""
        return torch.cat(arrays, dim=axis)

    def stack(self, arrays: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
        """Stack tensors along new axis."""
        return torch.stack(arrays, dim=axis)

    def slice_last_dim(self, array: torch.Tensor, dim: int) -> torch.Tensor:
        """Slice tensor to specific dimension along last axis."""
        if array.dim() == 1:
            return array[:dim]
        elif array.dim() == 2:
            return array[:, :dim]
        else:
            return array[..., :dim]

    def to_numpy(self, array: torch.Tensor) -> np.ndarray:
        """Convert tensor to NumPy array."""
        # Move to CPU first if on GPU
        if array.device.type in ['cuda', 'mps']:
            array = array.cpu()
        return array.detach().numpy()

    def from_numpy(self, array: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to tensor."""
        return torch.from_numpy(array).to(self.device)

    def save(self, array: torch.Tensor, filepath: str) -> None:
        """Save tensor to file (as NumPy for portability)."""
        np_array = self.to_numpy(array)
        if filepath.endswith('.npy'):
            np.save(filepath, np_array)
        elif filepath.endswith('.npz'):
            np.savez_compressed(filepath, data=np_array)
        else:
            # Default to .npy
            np.save(filepath, np_array)

    def load(self, filepath: str) -> torch.Tensor:
        """Load tensor from file."""
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            np_array = data['data']
        else:
            np_array = np.load(filepath)

        return self.from_numpy(np_array)

    def get_memory_usage(self, array: torch.Tensor) -> int:
        """Get memory usage in bytes."""
        return array.element_size() * array.nelement()

    def get_shape(self, array: torch.Tensor) -> Tuple[int, ...]:
        """Get tensor shape."""
        return tuple(array.shape)

    def get_dtype(self, array: torch.Tensor) -> str:
        """Get tensor dtype as string."""
        dtype_map = {
            torch.float32: 'float32',
            torch.float64: 'float64',
            torch.float16: 'float16',
            torch.int32: 'int32',
            torch.int64: 'int64',
            torch.uint8: 'uint8',
        }
        return dtype_map.get(array.dtype, str(array.dtype))

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to PyTorch dtype."""
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
        }
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype_map[dtype_str]
