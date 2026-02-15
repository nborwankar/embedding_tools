# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2025-12-29

## Repository Overview

`embedding_tools` is a production-ready Python library providing backend-agnostic array operations for embedding experiments. Extracted from the kb_tree_matryoshka research project, it offers seamless switching between NumPy (CPU), MLX (Apple Silicon GPU), JAX (GPU/TPU with JIT), and PyTorch (CUDA/MPS/CPU) with zero code changes.

**Key Features**:
- Backend abstraction for cross-platform embedding workflows
- Memory-safe storage with configurable limits (`EmbeddingStore`)
- Configuration versioning via SHA-256 hashing for reproducibility
- Optimized for Matryoshka embeddings (dimension slicing)
- GPU acceleration: MLX (M-series Macs), JAX (GPU/TPU with JIT compilation), and PyTorch (CUDA/MPS)

## Development Environment

**Current Setup**: Conda environment `embedding_tools` with Python 3.11.14

```bash
# Activate environment
conda activate embedding_tools

# Install in development mode
pip install -e ".[dev]"

# Install with all backends
pip install -e ".[all]"
```

**Backend Dependencies**:
- NumPy backend: Always available (required dependency)
- MLX backend: `pip install ".[mlx]"` (Apple Silicon only)
- JAX backend: `pip install ".[jax]"` (cross-platform GPU/TPU)
- PyTorch backend: `pip install ".[torch]"` (cross-platform)

## Common Commands

### Testing

```bash
# Run all tests (52 core tests + 23 JAX tests + 7 PyTorch tests = 75 total)
pytest tests/ -v

# Run specific test module
pytest tests/test_arrays.py -v
pytest tests/test_memory.py -v
pytest tests/test_config.py -v
pytest tests/test_jax_backend.py -v
pytest tests/test_torch_backend.py -v

# Quick installation validation
python validate.py
pytest tests/test_installation.py -v
```

### Code Quality

```bash
# Format code
black embedding_tools/ tests/ examples/
isort embedding_tools/ tests/ examples/

# Lint
flake8 embedding_tools/

# Format + lint (combined)
black . && isort . && flake8 embedding_tools/
```

### Running Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Device detection workflow
python examples/device_detection_workflow.py

# Configuration-driven backend selection
python examples/config_driven_backend.py

# Device info utility
python -m embedding_tools.utils.device_detection
```

## Architecture

### Backend Abstraction Pattern

The library uses abstract base class polymorphism to provide a unified interface across four array backends:

1. **Base Layer** (`arrays/base.py`):
   - `ArrayBackend` abstract class defines 17 operations
   - `get_backend(name, device)` factory function with auto-detection
   - Auto-detection priority: MLX (Mac) → JAX → PyTorch → NumPy

2. **Backend Implementations**:
   - `NumpyBackend`: Pure NumPy (CPU fallback)
   - `MLXBackend`: Apple Silicon GPU via MLX framework
   - `JAXBackend`: GPU/TPU/CPU with JIT compilation via JAX/XLA
   - `TorchBackend`: PyTorch with device selection (CUDA/MPS/CPU)

3. **Key Design Decision**: All backends implement identical interfaces, enabling code portability:
   ```python
   # Same code works on Mac (MLX/JAX), Linux (CUDA/JAX), or CPU (NumPy)
   backend = get_backend()  # Auto-detects best option
   embeddings = backend.create_array(data)
   sims = backend.cosine_similarity(query, embeddings)
   ```

### Memory Management (`EmbeddingStore`)

Multi-dimensional embedding storage with memory limits:
- Stores embeddings keyed by dimension (e.g., 32D, 128D, 1024D)
- Enforces `max_memory_gb` limit before adding new embeddings
- Supports Matryoshka slicing: `slice_to_dimension(1024, 128)`
- Backend-agnostic: Works with any ArrayBackend
- Metadata tracking: text_ids, labels, custom metadata per dimension

**Why EmbeddingStore exists**: Large-scale embedding experiments (MS MARCO, WordNet) can easily exceed memory. This provides controlled memory usage with automatic dimension management for Matryoshka experiments.

### Configuration Versioning

SHA-256 hashing for experiment reproducibility:
- `compute_config_hash(config_dict)` → 16-char hex
- `compute_param_hash(**kwargs)` → 16-char hex (convenience)
- Used for cache invalidation and result tracking

**Use case**: Hash experiment parameters (model, batch_size, dimensions) to create unique cache keys. If config changes, hash changes → new experiment run.

### Device Detection Utilities

Cross-platform backend and device detection (`utils/device_detection.py`):
- `detect_best_backend()` → 'mlx' | 'torch' | 'numpy'
- `detect_best_device()` → 'cuda' | 'mps' | 'cpu' | None
- `get_device_info()` → detailed hardware/backend report
- `detect_backend_with_fallback(prefer_performance=True)` → strategic selection

**Performance vs Consistency Trade-off**:
- `prefer_performance=True`: MLX > PyTorch MPS (faster on Mac)
- `prefer_performance=False`: PyTorch > MLX (cross-platform consistency)

## Backend-Specific Notes

### MLX Backend (Apple Silicon)

- **20-40% faster** than PyTorch MPS on M-series Macs
- Uses unified memory (shares with system RAM)
- File I/O: Converts to NumPy (no native MLX format)
- Best for: Mac-only experiments prioritizing speed

**Limitation**: macOS-only, not available on Linux/Windows

### PyTorch Backend (Cross-Platform)

- **Device auto-detection**: CUDA → MPS → CPU
- Explicit device configuration: `get_backend('torch', device='cuda')`
- **Critical fix**: Uses `from __future__ import annotations` to prevent import errors when PyTorch unavailable
- Best for: Cross-platform development, Linux production (CUDA)

**Device mapping**:
- `device='cuda'`: NVIDIA GPUs (Linux/Windows)
- `device='mps'`: Apple Silicon GPU (macOS)
- `device='cpu'`: Universal fallback

### JAX Backend (Cross-Platform with JIT)

- **JIT compilation via XLA**: 5-10x faster on repeated operations
- **Device auto-detection**: GPU/TPU → CPU
- Explicit device configuration: `get_backend('jax', device='gpu')`
- **Critical feature**: Pre-compiled kernels for cosine similarity
- Best for: Research workflows, repeated operations, TPU support

**Device mapping**:
- `device='gpu'`: GPU acceleration (Metal/CUDA/ROCm auto-detected)
- `device='cpu'`: CPU fallback (all platforms)
- `device=None`: Auto-detection (recommended)

**Performance characteristics**:
- First call: ~70ms (includes JIT compilation)
- Subsequent calls: ~0.05ms (uses compiled kernel)
- **Speedup: ~1500x** after JIT warmup
- Best for batch processing and research

**Technical details**:
- Uses `from __future__ import annotations` for safe imports
- Random number generation uses fixed PRNG key for reproducibility
- File I/O converts to NumPy format (no native JAX serialization)
- Normalize function not JIT-compiled (dynamic axis parameter)

**Installation**:
- macOS: `pip install embedding_tools[jax]` (includes jax-metal for Apple Silicon)
- Linux: `pip install embedding_tools[jax]` (CUDA via separate install)

### NumPy Backend (Universal)

- Always available (required dependency)
- CPU-only (no GPU acceleration)
- Baseline performance (1x)
- Best for: Testing, CI/CD, CPU-only environments

## Common Development Patterns

### Adding New Backend

**JAX backend has been implemented** (December 2024) - see `DONE.md` for implementation details.

To add another backend (e.g., TensorFlow):

1. Create `arrays/tf_backend.py` implementing `ArrayBackend`
2. Implement all 17 abstract methods
3. Add optional dependency to `pyproject.toml`: `tensorflow = ["tensorflow>=2.0.0"]`
4. Update `arrays/__init__.py` with import/detection:
   ```python
   try:
       from .tf_backend import TensorFlowBackend
       TF_AVAILABLE = True
   except ImportError:
       TF_AVAILABLE = False
   ```
5. Update `get_backend()` factory in `base.py`
6. Add tests in `tests/test_tf_backend.py`
7. Update docs: README.md, USAGE_EXAMPLES.md, CHANGELOG.md, CLAUDE.md

### Adding New Operations

To extend `ArrayBackend` with new operations:

1. Add abstract method to `ArrayBackend` class in `arrays/base.py`
2. Implement in all four backends: NumPy, MLX, JAX, PyTorch
3. Add tests to `tests/test_arrays.py` for each backend
4. Document in README.md API Reference section

**IMPORTANT**: All backends must maintain interface parity. Adding to one requires adding to all.

## Test Run Logging

**MANDATORY: Every test run must be logged.**

1. **Raw output** saved to `docs/test_runs/YYYY-MM-DD_p{phase}_t{task}_{description}.txt`
2. **Summary row** appended to `docs/TEST_LOG.md` in the table format:
   ```
   | datetime | project | phase | task | command | passed | failed | skipped | deselected | duration_sec | context | commit | raw_output |
   ```

The table is designed for future import into SQLite/PostgreSQL for data mining (test count trends, duration regressions, failure rate by phase). Each row is one logical database record.

See `docs/TEST_LOG.md` for the current log and `docs/test_runs/` for raw outputs.

---

## Testing Strategy

### Test Organization

- `test_installation.py`: Post-install validation (16 tests)
- `test_arrays.py`: Backend operations, cross-conversion (19 tests)
- `test_memory.py`: EmbeddingStore functionality (10 tests)
- `test_config.py`: Configuration hashing (7 tests)
- `test_jax_backend.py`: JAX-specific scenarios (23 tests)
- `test_torch_backend.py`: PyTorch-specific scenarios (7 tests)

### Running Subset Tests

```bash
# Test only NumPy backend
pytest tests/test_arrays.py::TestNumpyBackend -v

# Test specific operation
pytest tests/test_arrays.py::TestNumpyBackend::test_cosine_similarity -v

# Test EmbeddingStore memory limits
pytest tests/test_memory.py::TestEmbeddingStore::test_memory_limit -v
```

### Writing New Tests

- Use `pytest` framework (configured in `pyproject.toml`)
- Mark backend-specific tests with conditional skips:
  ```python
  @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
  def test_mlx_feature():
      ...
  ```
- Test all backends when adding new operations
- Include memory tracking tests for EmbeddingStore changes

## GPU Acceleration on Apple Silicon

**CRITICAL**: Always use MPS-first device detection pattern:

```python
# CORRECT - Checks MPS first (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

**DO NOT** use old CUDA-only pattern:
```python
# WRONG - Ignores MPS on M2 Macs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Why this matters**: The TorchBackend auto-detection follows this pattern. When users don't specify device explicitly, MPS detection comes first to leverage Apple Silicon GPU acceleration.

## Package Distribution

### Local Installation

```bash
# Development mode (changes reflected immediately)
pip install -e .

# With specific extras
pip install -e ".[mlx]"      # Apple Silicon support
pip install -e ".[torch]"    # PyTorch support
pip install -e ".[all]"      # All backends
pip install -e ".[dev]"      # Development tools
```

### Version Management

- Version defined in `pyproject.toml` and `__init__.py`
- Update both files together: `__version__ = '0.1.0'`
- Follow semantic versioning (MAJOR.MINOR.PATCH)

### Future PyPI Publication

Not yet published. To prepare:
1. Verify `pyproject.toml` metadata complete
2. Build distributions: `python -m build`
3. Test on TestPyPI first
4. Upload: `twine upload dist/*`

## Known Issues and Fixes

### PyTorch Import Error (RESOLVED)

**Issue**: `NameError: name 'torch' is not defined` when PyTorch unavailable but TorchBackend imported.

**Fix**: Added `from __future__ import annotations` to `torch_backend.py` (line 1). This defers type hint evaluation, allowing the module to import even when torch unavailable.

**Impact**: NumPy and MLX backends work even with broken/missing PyTorch installation.

### PyTorch Dylib Corruption (RESOLVED)

**Issue**: `Library not loaded: @rpath/libtorch_cpu.dylib` on some installations.

**Fix**: Use clean conda environment instead of pip in base Python:
```bash
conda create -n embedding_tools python=3.11 -y
conda activate embedding_tools
pip install -e ".[all]"
```

**Documented in**: `PYTORCH_FIX.md`

## Related Documentation

- `README.md`: User-facing documentation, API reference, installation
- `USAGE_EXAMPLES.md`: 8 practical workflows and patterns
- `MLX_VS_MPS.md`: Performance comparison (MLX vs PyTorch MPS)
- `FALLBACK_STRATEGY.md`: Backend selection strategies
- `JAX_PLAN.md`: Future JAX backend implementation plan
- `EXPT_VERSIONING.md`: Configuration versioning best practices
- `DONE.md`: Complete development history and decisions
- `CHANGELOG.md`: Version history and release notes

## Integration with kb_tree_matryoshka

This library was extracted from the kb_tree_matryoshka research project. When working with that project:

1. Install embedding_tools as dependency: `pip install -e ../embedding_tools/`
2. Replace ad-hoc numpy operations with backend abstraction
3. Use `EmbeddingStore` for multi-dimensional embeddings (32D, 128D, 512D, 1024D)
4. Use `compute_param_hash()` for experiment cache keys
5. Enable MLX backend for 20-40% speedup on M2 Mac

**Import pattern in kb_tree experiments**:
```python
from embedding_tools import get_backend, EmbeddingStore, compute_param_hash

# Auto-detect best backend for current platform
backend = get_backend()

# Or explicit for production consistency
if platform == 'mac_dev':
    store = EmbeddingStore(backend='mlx', max_memory_gb=20)
elif platform == 'linux_prod':
    store = EmbeddingStore(backend='torch', max_memory_gb=40, device='cuda')
```

## Performance Benchmarks

From `MLX_VS_MPS.md`:

| Backend | Device | Relative Speed | Use Case |
|---------|--------|----------------|----------|
| NumPy | CPU | 1x (baseline) | Universal fallback |
| PyTorch | CPU | 1.2x | Cross-platform consistency |
| PyTorch | MPS | 2-3x | Mac with PyTorch ecosystem |
| MLX | Apple GPU | 3-5x | Mac-only, best performance |
| PyTorch | CUDA | 4-10x | Linux production (NVIDIA) |

**Recommendation**: Use MLX for Mac development, PyTorch with CUDA for Linux production.
