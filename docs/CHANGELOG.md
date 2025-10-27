# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-26

### Added

**Initial Release**
- Cross-platform backend abstraction (NumPy, MLX, PyTorch)
- `NumpyBackend`: CPU-based operations (universal fallback)
- `MLXBackend`: Apple Silicon GPU acceleration (20-40% faster than PyTorch MPS)
- `TorchBackend`: CUDA/MPS/CPU support with auto-detection
- `EmbeddingStore`: Memory-managed storage for multi-dimensional embeddings
- Configuration versioning with SHA-256 hashing
- Cosine similarity and nearest neighbor search
- Dimension slicing for Matryoshka embeddings

**Testing**
- 52 comprehensive tests covering all backends
- Installation validation script (`validate.py`)
- Example code in `examples/` directory

**Documentation**
- Complete README with usage examples
- `USAGE_EXAMPLES.md`: 8 practical workflows
- `MLX_VS_MPS.md`: Performance comparison guide
- `FALLBACK_STRATEGY.md`: Backend fallback configuration
- `JAX_PLAN.md`: Future JAX backend implementation plan
- `PYTORCH_FIX.md`: Type hint bug fix documentation

### Fixed

**PyTorch Type Hint Bug**
- Added `from __future__ import annotations` to `torch_backend.py`
- Fixes `NameError: name 'torch' is not defined` when PyTorch unavailable
- Allows NumPy/MLX backends to work even with broken PyTorch installation

### Dependencies

**Required**
- numpy>=1.21.0

**Optional**
- mlx>=0.0.7 (Apple Silicon GPU support)
- torch>=2.0.0 (CUDA/MPS/CPU support)

**Development**
- pytest>=7.0.0
- pytest-cov>=4.0.0
- black>=22.0.0
- isort>=5.10.0
- flake8>=4.0.0

---

## Future Releases

### Planned for 0.2.0

**JAX Backend** (See JAX_PLAN.md)
- JIT-compiled operations for 5-10x speedup
- GPU/TPU support (Metal/CUDA/ROCm)
- Expected implementation time: 6-8 hours

**Performance Improvements**
- Batch processing optimizations
- Memory pooling for reduced allocation overhead

**Additional Backends**
- TensorFlow support (lower priority)

### Under Consideration

- PyPI publication
- CI/CD integration
- Expanded test coverage
- Benchmark suite

---

## Version History

- **0.1.0** (2025-10-26) - Initial release

---

[0.1.0]: https://github.com/nborwankar/embedding_tools/releases/tag/v0.1.0
