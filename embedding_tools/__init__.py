"""embedding_tools: Utilities for embedding experiments with cross-platform array support.

This library provides:
- Backend-agnostic array operations (NumPy, MLX, PyTorch)
- Memory-safe embedding storage
- Configuration versioning
- Similarity search utilities
- Device detection and auto-configuration
"""

__version__ = "0.1.2"

from .arrays import (
    ArrayBackend,
    get_backend,
    NumpyBackend,
    MLXBackend,
    TorchBackend,
    MLX_AVAILABLE,
    TORCH_AVAILABLE,
)

from .memory import EmbeddingStore

from .config import compute_config_hash, compute_param_hash

from .utils import detect_best_backend, detect_best_device, get_device_info

__all__ = [
    # Version
    "__version__",
    # Array backends
    "ArrayBackend",
    "get_backend",
    "NumpyBackend",
    "MLXBackend",
    "TorchBackend",
    "MLX_AVAILABLE",
    "TORCH_AVAILABLE",
    # Memory management
    "EmbeddingStore",
    # Configuration
    "compute_config_hash",
    "compute_param_hash",
    # Device detection
    "detect_best_backend",
    "detect_best_device",
    "get_device_info",
]
