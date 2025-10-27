"""Array backend system for cross-platform embedding operations."""

from .base import ArrayBackend, get_backend
from .numpy_backend import NumpyBackend

try:
    from .mlx_backend import MLXBackend
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    MLXBackend = None

try:
    from .torch_backend import TorchBackend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchBackend = None

__all__ = [
    'ArrayBackend',
    'get_backend',
    'NumpyBackend',
    'MLXBackend',
    'TorchBackend',
    'MLX_AVAILABLE',
    'TORCH_AVAILABLE',
]
