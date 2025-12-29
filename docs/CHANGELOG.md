# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**JAX Backend Support** (December 2025)
- `JAXBackend`: GPU/TPU/CPU support with JIT compilation via JAX/XLA
- JIT-compiled cosine similarity kernel for 2-3x speedup on repeated operations
- Cross-platform GPU support: Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD)
- Auto-detection priority updated: MLX → JAX → PyTorch → NumPy
- Device configuration: `device='gpu'` or `device='cpu'`
- Pre-compiled kernels in `__init__` for optimal performance
- Fixed PRNG key for reproducible random number generation

**Testing**
- 23 comprehensive JAX backend tests (all passing)
- Total test suite: 75 tests (71 passing on Linux, 75 on macOS with MLX)
- JIT compilation speedup verification test (~1500x speedup after warmup)
- Large array stress tests
- EmbeddingStore integration tests with JAX
- Device specification tests (CPU/GPU)

**Documentation**
- `TESTING.md`: Comprehensive testing guide with git clone instructions
- Updated README.md with JAX installation and usage examples
- Updated CLAUDE.md with JAX backend information
- Updated DONE.md with complete JAX implementation session
- JAX backend comparison table in all relevant docs

**Performance Characteristics**
- First call: ~70ms (includes JIT compilation overhead)
- Subsequent calls: ~0.05ms (uses compiled kernel)
- 5-10x faster than NumPy on repeated operations
- Best for: Research workflows, batch processing, repeated similarity searches

### Changed
- Auto-detection priority now: MLX → JAX → PyTorch → NumPy
- Updated backend count from 3 to 4 in all documentation
- Keywords in pyproject.toml: Added "jax" and "pytorch"

### Dependencies

**New Optional**
- jax>=0.4.0 (cross-platform)
- jax-metal>=0.1.0 (macOS Apple Silicon only)

**Installation**
- `pip install embedding_tools[jax]` - JAX support
- `pip install embedding_tools[all]` - All backends (now includes JAX)

---

## [0.1.1] - 2025-10-27

### Fixed
- MLX backend import error when MLX not installed
- Added `from __future__ import annotations` to `mlx_backend.py`

### Changed
- License format updated to SPDX identifier
- README updated with PyPI installation instructions
- PyPI badges added to README

---

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

### Planned for Next Release

**Performance Improvements**
- Batch processing optimizations
- Memory pooling for reduced allocation overhead
- Multi-device support for JAX (shard across GPUs)
- Advanced JIT optimization with static arguments

**Additional Features**
- Additional similarity metrics (euclidean, manhattan)
- Sparse array support
- GPU memory pooling

**Additional Backends**
- TensorFlow support (if requested)

### Under Consideration

- CI/CD integration (GitHub Actions)
- Expanded test coverage (target 95%+)
- Comprehensive benchmark suite
- Cross-backend performance comparison tools

---

## Version History

- **Unreleased** - JAX backend implementation
- **0.1.1** (2025-10-27) - PyPI release with MLX import fix
- **0.1.0** (2025-10-26) - Initial release

---

[Unreleased]: https://github.com/nborwankar/embedding_tools/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nborwankar/embedding_tools/releases/tag/v0.1.1
[0.1.0]: https://github.com/nborwankar/embedding_tools/releases/tag/v0.1.0
