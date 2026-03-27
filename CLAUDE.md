# CLAUDE.md

**Last Updated**: 2025-12-29

## Repository Overview

`embedding_tools` is a production-ready Python library providing backend-agnostic array operations for embedding experiments. Extracted from the kb_tree_matryoshka research project.

**Key Features**: Backend abstraction (NumPy/MLX/JAX/PyTorch), memory-safe `EmbeddingStore`, config versioning via SHA-256 hashing, Matryoshka dimension slicing, GPU acceleration (MLX, JAX JIT, PyTorch MPS/CUDA).

**Conda Environment**: `embedding_tools` (Python 3.11.14)

See global CLAUDE.md for conda activation, MPS device detection, git practices, batch edit rules.

## Architecture

### Backend Abstraction Pattern

```
┌─────────────────────────────────────────┐
│  ArrayBackend (abstract, 17 operations) │
│  get_backend(name, device) factory      │
│  Auto-detect: MLX → JAX → PyTorch → NumPy│
├─────────────┬───────────┬───────────────┤
│ NumpyBackend│ MLXBackend│ JAXBackend    │ TorchBackend
│ (CPU)       │ (Apple GPU)│ (GPU/TPU+JIT)│ (CUDA/MPS/CPU)
└─────────────┴───────────┴───────────────┘
```

**Key Design Decision**: All backends implement identical interfaces, enabling code portability:
```python
backend = get_backend()  # Auto-detects best option
embeddings = backend.create_array(data)
sims = backend.cosine_similarity(query, embeddings)
```

### Memory Management (`EmbeddingStore`)

Multi-dimensional embedding storage with memory limits:
- Stores embeddings keyed by dimension (e.g., 32D, 128D, 1024D)
- Enforces `max_memory_gb` limit before adding new embeddings
- Supports Matryoshka slicing: `slice_to_dimension(1024, 128)`
- Backend-agnostic, metadata tracking per dimension

**Why EmbeddingStore exists**: Large-scale embedding experiments can easily exceed memory. Provides controlled usage with automatic dimension management for Matryoshka experiments.

### Configuration Versioning

- `compute_config_hash(config_dict)` / `compute_param_hash(**kwargs)` -> 16-char hex SHA-256
- Used for cache invalidation and experiment result tracking

### Device Detection Utilities (`utils/device_detection.py`)

- `detect_best_backend()` -> 'mlx' | 'torch' | 'numpy'
- `detect_best_device()` -> 'cuda' | 'mps' | 'cpu' | None
- `get_device_info()` -> detailed hardware/backend report
- `detect_backend_with_fallback(prefer_performance=True)` -> strategic selection

**Performance vs Consistency Trade-off**:
- `prefer_performance=True`: MLX > PyTorch MPS (faster on Mac)
- `prefer_performance=False`: PyTorch > MLX (cross-platform consistency)

## Backend-Specific Notes

### MLX (Apple Silicon)
- **20-40% faster** than PyTorch MPS on M-series Macs
- Uses unified memory, file I/O via NumPy conversion
- macOS-only

### PyTorch (Cross-Platform)
- Device auto-detection: CUDA -> MPS -> CPU
- Uses `from __future__ import annotations` to prevent import errors when PyTorch unavailable

### JAX (Cross-Platform with JIT)
- First call: ~70ms (JIT compilation), subsequent: ~0.05ms (**~1500x speedup**)
- Fixed PRNG key for reproducibility
- Normalize function not JIT-compiled (dynamic axis parameter)
- macOS install includes jax-metal for Apple Silicon

### NumPy (Universal)
- Always available, CPU-only, baseline performance (1x)

## Testing

### Test Organization (75 total)
- `test_installation.py` (16) — post-install validation
- `test_arrays.py` (19) — backend operations, cross-conversion
- `test_memory.py` (10) — EmbeddingStore functionality
- `test_config.py` (7) — configuration hashing
- `test_jax_backend.py` (23) — JAX-specific scenarios
- `test_torch_backend.py` (7) — PyTorch-specific scenarios

Test runs logged per project convention: raw output to `docs/test_runs/`, summary to `docs/TEST_LOG.md`.

## Package Distribution

- Version in `pyproject.toml` and `__init__.py` (keep in sync)
- Install extras: `pip install -e ".[mlx]"`, `".[jax]"`, `".[torch]"`, `".[all]"`, `".[dev]"`
- Not yet published to PyPI

## Known Issues (Resolved)

**PyTorch Import Error**: `from __future__ import annotations` in `torch_backend.py` defers type hints, allowing import when torch unavailable.

**PyTorch Dylib Corruption**: Use clean conda environment instead of pip in base Python. See `PYTORCH_FIX.md`.

## Integration with kb_tree_matryoshka

```python
from embedding_tools import get_backend, EmbeddingStore, compute_param_hash

backend = get_backend()  # Auto-detect
store = EmbeddingStore(backend='mlx', max_memory_gb=20)  # Mac dev
store = EmbeddingStore(backend='torch', max_memory_gb=40, device='cuda')  # Linux prod
```

## Performance Benchmarks

| Backend | Device | Relative Speed | Use Case |
|---------|--------|----------------|----------|
| NumPy | CPU | 1x (baseline) | Universal fallback |
| PyTorch | CPU | 1.2x | Cross-platform consistency |
| PyTorch | MPS | 2-3x | Mac with PyTorch ecosystem |
| MLX | Apple GPU | 3-5x | Mac-only, best performance |
| PyTorch | CUDA | 4-10x | Linux production (NVIDIA) |

**Recommendation**: MLX for Mac development, PyTorch+CUDA for Linux production.
