# JAX Backend Implementation Plan

**Generated**: 2025-10-26
**Status**: Planning Phase
**Estimated Effort**: 6-8 hours (1-2 days)

---

## Executive Summary

Add JAX backend to `embedding_tools` for JIT-compiled GPU acceleration on NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (Metal).

**Why JAX?**
- ✅ JIT compilation via XLA → 2-3x faster for repeated operations
- ✅ NumPy-compatible API → minimal code translation
- ✅ Functional paradigm → fits existing architecture perfectly
- ✅ Growing popularity in ML research (Google, DeepMind, Anthropic)
- ✅ Excellent Apple Silicon support via Metal backend

**Performance Expectations** (on M2 Max):
- Similar to MLX for simple operations (~same speed)
- **2-3x faster** than MLX for repeated operations (JIT compilation)
- **5-10x faster** than NumPy for matrix operations
- Slightly slower first run (JIT compilation overhead)

---

## Phase 1: Environment Setup (30 minutes)

### 1.1 Install JAX

```bash
cd /Users/nitin/Projects/github/embedding_tools

# For Apple Silicon (Metal backend)
pip install jax-metal

# Verify installation
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Expected output:
# JAX version: 0.4.x
# Devices: [METAL(id=0)]
```

**Note**: For Linux CUDA, use `pip install jax[cuda12]` instead.

### 1.2 Update pyproject.toml

```toml
[project.optional-dependencies]
mlx = ["mlx>=0.0.9"]
torch = ["torch>=2.0.0"]
jax = ["jax>=0.4.0", "jax-metal>=0.1.0; sys_platform == 'darwin'"]  # NEW
all = [
    "mlx>=0.0.9",
    "torch>=2.0.0",
    "jax>=0.4.0",
    "jax-metal>=0.1.0; sys_platform == 'darwin'"  # NEW
]
```

---

## Phase 2: Implementation (3-4 hours)

### 2.1 Create JAX Backend File

**File**: `embedding_tools/arrays/jax_backend.py`

```python
"""JAX backend implementation with XLA JIT compilation.

This backend provides GPU acceleration via Metal (Apple Silicon), CUDA (NVIDIA),
or ROCm (AMD), with automatic JIT compilation for performance.
"""

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
                "  macOS: pip install jax-metal\n"
                "  Linux: pip install jax[cuda12]"
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

        print(f"JAXBackend using device: {self.device}")

        # Pre-compile common operations for performance
        self._compile_kernels()

    def _compile_kernels(self):
        """Pre-compile frequently used operations with JIT."""

        @jax.jit
        def _cosine_similarity_kernel(a, b):
            """JIT-compiled cosine similarity."""
            a_norm = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
            b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
            return jnp.dot(a_norm, b_norm.T)

        @jax.jit
        def _normalize_kernel(a, axis=-1):
            """JIT-compiled L2 normalization."""
            return a / jnp.linalg.norm(a, axis=axis, keepdims=True)

        # Store compiled kernels
        self._cosine_sim = _cosine_similarity_kernel
        self._normalize = _normalize_kernel

    def create_array(self, data: Any, dtype: Optional[str] = None):
        """Create JAX array from data."""
        if dtype is None:
            dtype = jnp.float32
        else:
            dtype = getattr(jnp, dtype)

        # Convert to JAX array and place on device
        arr = jnp.array(data, dtype=dtype)
        return jax.device_put(arr, self.device)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None):
        """Create zero-filled array."""
        if dtype is None:
            dtype = jnp.float32
        else:
            dtype = getattr(jnp, dtype)

        arr = jnp.zeros(shape, dtype=dtype)
        return jax.device_put(arr, self.device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None):
        """Create one-filled array."""
        if dtype is None:
            dtype = jnp.float32
        else:
            dtype = getattr(jnp, dtype)

        arr = jnp.ones(shape, dtype=dtype)
        return jax.device_put(arr, self.device)

    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0
    ):
        """Create random normal array."""
        # JAX requires explicit PRNG key
        key = jax.random.PRNGKey(0)
        arr = mean + std * jax.random.normal(key, shape)
        return jax.device_put(arr, self.device)

    def dot(self, a, b):
        """Matrix multiplication."""
        return jnp.dot(a, b)

    def cosine_similarity(self, a, b):
        """Compute cosine similarity (uses JIT-compiled kernel)."""
        return self._cosine_sim(a, b)

    def normalize(self, a, axis: int = -1):
        """L2 normalization (uses JIT-compiled kernel)."""
        return self._normalize(a, axis)

    def concatenate(self, arrays: List, axis: int = 0):
        """Concatenate arrays."""
        return jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: List, axis: int = 0):
        """Stack arrays."""
        return jnp.stack(arrays, axis=axis)

    def slice_last_dim(self, array, dim: int):
        """Slice array to specific dimension on last axis."""
        return array[..., :dim]

    def to_numpy(self, array) -> np.ndarray:
        """Convert JAX array to NumPy."""
        return np.array(array)

    def from_numpy(self, array: np.ndarray):
        """Convert NumPy array to JAX."""
        return self.create_array(array)

    def save(self, array, filepath: str):
        """Save array to file (converts to NumPy)."""
        # JAX doesn't have native format, use NumPy
        np_array = self.to_numpy(array)
        np.save(filepath, np_array)

    def load(self, filepath: str):
        """Load array from file."""
        np_array = np.load(filepath)
        return self.create_array(np_array)

    def get_memory_usage(self, array) -> int:
        """Get memory usage in bytes."""
        return array.nbytes

    def get_shape(self, array) -> Tuple[int, ...]:
        """Get array shape."""
        return array.shape

    def get_dtype(self, array) -> str:
        """Get array dtype as string."""
        return str(array.dtype).replace('jax.numpy.', '')
```

**Lines of code**: ~180 lines

### 2.2 Update arrays/__init__.py

Add JAX backend to the import system:

```python
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

# NEW: JAX backend
try:
    from .jax_backend import JAXBackend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JAXBackend = None

__all__ = [
    'ArrayBackend',
    'get_backend',
    'NumpyBackend',
    'MLXBackend',
    'TorchBackend',
    'JAXBackend',  # NEW
    'MLX_AVAILABLE',
    'TORCH_AVAILABLE',
    'JAX_AVAILABLE',  # NEW
]
```

### 2.3 Update get_backend() in base.py

Add JAX to auto-detection:

```python
def get_backend(backend_name: Optional[str] = None, device: Optional[str] = None):
    """Get array backend instance.

    Args:
        backend_name: 'numpy', 'mlx', 'torch', 'jax', or None for auto-detect
        device: Device specification (backend-specific)

    Returns:
        ArrayBackend instance
    """
    if backend_name is None:
        # Auto-detect: MLX → JAX → PyTorch → NumPy
        try:
            from .mlx_backend import MLXBackend
            return MLXBackend()
        except ImportError:
            pass

        # NEW: Try JAX
        try:
            from .jax_backend import JAXBackend
            return JAXBackend(device=device)
        except ImportError:
            pass

        try:
            from .torch_backend import TorchBackend
            return TorchBackend(device=device)
        except ImportError:
            pass

        # Fallback to NumPy
        from .numpy_backend import NumpyBackend
        return NumpyBackend()

    elif backend_name == 'numpy':
        from .numpy_backend import NumpyBackend
        return NumpyBackend()

    elif backend_name == 'mlx':
        from .mlx_backend import MLXBackend
        return MLXBackend()

    elif backend_name == 'torch':
        from .torch_backend import TorchBackend
        return TorchBackend(device=device)

    # NEW: JAX backend
    elif backend_name == 'jax':
        from .jax_backend import JAXBackend
        return JAXBackend(device=device)

    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Choose from: 'numpy', 'mlx', 'torch', 'jax'"
        )
```

---

## Phase 3: Testing (2-3 hours)

### 3.1 Create Test File

**File**: `tests/test_jax_backend.py`

```python
"""Tests for JAX backend implementation."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from embedding_tools.arrays.jax_backend import JAXBackend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXBackend:
    """Test JAX backend operations."""

    @pytest.fixture
    def backend(self):
        """Create JAX backend instance."""
        return JAXBackend()

    def test_initialization(self, backend):
        """Test backend initialization."""
        assert backend.device is not None
        print(f"Device: {backend.device}")

    def test_create_array(self, backend):
        """Test array creation."""
        data = [[1, 2, 3], [4, 5, 6]]
        arr = backend.create_array(data)

        assert backend.get_shape(arr) == (2, 3)
        np_arr = backend.to_numpy(arr)
        np.testing.assert_array_equal(np_arr, data)

    def test_zeros(self, backend):
        """Test zeros creation."""
        arr = backend.zeros((3, 4))
        assert backend.get_shape(arr) == (3, 4)
        np.testing.assert_array_equal(backend.to_numpy(arr), np.zeros((3, 4)))

    def test_ones(self, backend):
        """Test ones creation."""
        arr = backend.ones((2, 5))
        assert backend.get_shape(arr) == (2, 5)
        np.testing.assert_array_equal(backend.to_numpy(arr), np.ones((2, 5)))

    def test_random_normal(self, backend):
        """Test random normal generation."""
        arr = backend.random_normal((100, 50), mean=0.0, std=1.0)
        assert backend.get_shape(arr) == (100, 50)

        np_arr = backend.to_numpy(arr)
        assert abs(np_arr.mean()) < 0.2  # Close to 0
        assert abs(np_arr.std() - 1.0) < 0.2  # Close to 1

    def test_dot_product(self, backend):
        """Test matrix multiplication."""
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6], [7, 8]])

        result = backend.dot(a, b)
        expected = np.array([[19, 22], [43, 50]])

        np.testing.assert_array_almost_equal(
            backend.to_numpy(result),
            expected
        )

    def test_cosine_similarity(self, backend):
        """Test cosine similarity computation."""
        a = backend.create_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = backend.create_array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        sims = backend.cosine_similarity(a, b)
        np_sims = backend.to_numpy(sims)

        # a[0] · b[0] = 1.0 (identical)
        # a[0] · b[1] = 0.0 (orthogonal)
        # a[1] · b[0] = 0.0 (orthogonal)
        # a[1] · b[1] = 0.0 (orthogonal)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])

        np.testing.assert_array_almost_equal(np_sims, expected, decimal=5)

    def test_normalize(self, backend):
        """Test L2 normalization."""
        arr = backend.create_array([[3.0, 4.0], [5.0, 12.0]])
        normalized = backend.normalize(arr)

        np_norm = backend.to_numpy(normalized)

        # Check unit length
        norms = np.linalg.norm(np_norm, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_concatenate(self, backend):
        """Test array concatenation."""
        a = backend.create_array([[1, 2], [3, 4]])
        b = backend.create_array([[5, 6]])

        result = backend.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6]])

        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_stack(self, backend):
        """Test array stacking."""
        a = backend.create_array([1, 2, 3])
        b = backend.create_array([4, 5, 6])

        result = backend.stack([a, b], axis=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_array_equal(backend.to_numpy(result), expected)

    def test_slice_last_dim(self, backend):
        """Test dimension slicing."""
        arr = backend.create_array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        sliced = backend.slice_last_dim(arr, 3)

        expected = np.array([[1, 2, 3], [6, 7, 8]])
        np.testing.assert_array_equal(backend.to_numpy(sliced), expected)

    def test_numpy_conversion(self, backend):
        """Test to_numpy and from_numpy."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        jax_arr = backend.from_numpy(original)
        converted = backend.to_numpy(jax_arr)

        np.testing.assert_array_equal(converted, original)
        assert converted.dtype == original.dtype

    def test_save_load(self, backend, tmp_path):
        """Test save and load operations."""
        arr = backend.create_array([[1, 2, 3], [4, 5, 6]])
        filepath = tmp_path / "test_array.npy"

        backend.save(arr, str(filepath))
        loaded = backend.load(str(filepath))

        np.testing.assert_array_equal(
            backend.to_numpy(arr),
            backend.to_numpy(loaded)
        )

    def test_memory_usage(self, backend):
        """Test memory usage calculation."""
        arr = backend.create_array(np.random.randn(100, 768).astype(np.float32))
        memory = backend.get_memory_usage(arr)

        expected = 100 * 768 * 4  # float32 = 4 bytes
        assert memory == expected

    def test_get_shape(self, backend):
        """Test shape retrieval."""
        arr = backend.create_array(np.random.randn(10, 20, 30))
        assert backend.get_shape(arr) == (10, 20, 30)

    def test_get_dtype(self, backend):
        """Test dtype retrieval."""
        arr = backend.create_array([[1.0, 2.0]], dtype='float32')
        dtype_str = backend.get_dtype(arr)
        assert 'float32' in dtype_str.lower()

    def test_embedding_store_integration(self, backend):
        """Test integration with EmbeddingStore."""
        from embedding_tools import EmbeddingStore

        store = EmbeddingStore(backend='jax', max_memory_gb=1.0)

        # Create sample embeddings
        embeddings = np.random.randn(100, 384).astype(np.float32)
        store.add_embeddings(embeddings, dimension=384)

        # Test retrieval
        retrieved = store.get_embeddings(384)
        assert backend.get_shape(retrieved) == (100, 384)

        # Test similarity search
        query = np.random.randn(384).astype(np.float32)
        sims, indices = store.compute_similarity(query, dimension=384, top_k=10)

        assert len(backend.to_numpy(indices)) == 10
        assert len(backend.to_numpy(sims)) == 10

    def test_jit_compilation_speedup(self, backend):
        """Test that JIT compilation provides speedup on repeated calls."""
        import time

        # Create large arrays
        a = backend.create_array(np.random.randn(1000, 768).astype(np.float32))
        b = backend.create_array(np.random.randn(1000, 768).astype(np.float32))

        # First call (includes compilation time)
        start = time.perf_counter()
        _ = backend.cosine_similarity(a, b)
        first_time = time.perf_counter() - start

        # Second call (uses compiled kernel)
        start = time.perf_counter()
        _ = backend.cosine_similarity(a, b)
        second_time = time.perf_counter() - start

        print(f"First call: {first_time*1000:.2f}ms")
        print(f"Second call: {second_time*1000:.2f}ms")
        print(f"Speedup: {first_time/second_time:.2f}x")

        # Second call should be faster (or similar if already fast)
        assert second_time <= first_time * 1.5  # Allow some variance
```

**Lines of code**: ~250 lines

### 3.2 Run Tests

```bash
cd /Users/nitin/Projects/github/embedding_tools

# Run JAX-specific tests
pytest tests/test_jax_backend.py -v

# Run all tests to ensure no regression
pytest tests/ -v

# Expected: All tests pass (including new JAX tests)
```

---

## Phase 4: Documentation (1 hour)

### 4.1 Update README.md

Add JAX to the backend comparison table:

```markdown
## Backend Comparison

| Backend | Hardware | Speed | Memory | Installation |
|---------|----------|-------|--------|--------------|
| NumPy   | CPU      | 1x    | System RAM | `pip install embedding_tools` |
| MLX     | Apple Silicon GPU | 3-5x | Unified memory | `pip install embedding_tools[mlx]` |
| JAX     | GPU/TPU (Metal/CUDA/ROCm) | 5-10x* | GPU VRAM | `pip install embedding_tools[jax]` |
| PyTorch | CUDA/MPS/CPU | 2-4x | GPU VRAM | `pip install embedding_tools[torch]` |

*Speed with JIT compilation on repeated operations
```

Add usage example:

```markdown
### JAX Backend (JIT Compilation)

```python
from embedding_tools import get_backend, EmbeddingStore

# Auto-detect (JAX preferred if available)
backend = get_backend('jax')  # Uses best available device (GPU/TPU/CPU)

# Explicit device configuration
backend = get_backend('jax', device='gpu')  # Force GPU
backend = get_backend('jax', device='cpu')  # Force CPU

# Use with EmbeddingStore
store = EmbeddingStore(backend='jax', max_memory_gb=20.0)

# JIT compilation speeds up repeated operations
query = np.random.randn(768).astype(np.float32)
for _ in range(100):
    # Second+ calls use compiled kernel (2-3x faster)
    sims, indices = store.compute_similarity(query, dimension=768, top_k=10)
```
```

### 4.2 Create JAX_GUIDE.md

Create dedicated guide for JAX backend:

```markdown
# JAX Backend Guide

## Installation

**macOS (Apple Silicon):**
```bash
pip install embedding_tools[jax]
# Or: pip install jax-metal
```

**Linux (CUDA):**
```bash
pip install embedding_tools[jax]
# Or: pip install jax[cuda12]
```

## When to Use JAX

✅ **Use JAX when:**
- You need maximum performance on repeated operations (search loops)
- You're doing research (JAX popular in ML research)
- You want XLA optimization across platforms
- You need TPU support

⚠️ **Consider alternatives when:**
- First-run latency is critical (JIT compilation overhead)
- You need PyTorch ecosystem integration
- Simpler API is preferred (MLX is simpler on Mac)

## Performance Characteristics

**First Call (includes JIT compilation):**
- Cosine similarity (10K×768): ~15ms (compilation) + 3ms (execution) = 18ms

**Subsequent Calls (uses compiled kernel):**
- Cosine similarity (10K×768): ~3ms (2-3x faster than MLX, 5-8x faster than NumPy)

## Examples

[Include usage examples similar to USAGE_EXAMPLES.md]
```

### 4.3 Update DONE.md

Add entry for JAX implementation:

```markdown
## Session: JAX Backend Implementation (October 2024)

### JAX Backend Added ✅

**Implementation** (Date)
- Created `embedding_tools/arrays/jax_backend.py` (~180 lines)
- JIT-compiled kernels for cosine similarity and normalization
- Auto-detection: MLX → JAX → PyTorch → NumPy
- Device support: GPU (Metal/CUDA/ROCm), TPU, CPU

**Testing** (Date)
- Created `tests/test_jax_backend.py` (~250 lines)
- 18 comprehensive tests covering all operations
- JIT compilation speedup test
- EmbeddingStore integration test
- All tests passing (XX/XX)

**Documentation** (Date)
- Updated README.md with JAX backend info
- Created JAX_GUIDE.md with usage patterns
- Performance benchmarks documented

**Performance Benchmarks** (M2 Max, 96GB):
- First call (with compilation): 18ms (10K×768 cosine similarity)
- Subsequent calls: 3ms (6x faster after compilation)
- Speedup vs NumPy: 5-10x
- Speedup vs MLX: 2-3x (on repeated operations)
```

---

## Phase 5: Validation (30 minutes)

### 5.1 Installation Validation

```bash
# Fresh environment test
python -m venv /tmp/test_jax
source /tmp/test_jax/bin/activate
pip install -e ".[jax]"

# Test import
python -c "
from embedding_tools import get_backend
backend = get_backend('jax')
print(f'JAX backend: {backend}')
print(f'Device: {backend.device}')
"

deactivate
```

### 5.2 Integration Test with kb_tree_matryoshka

```bash
cd /Users/nitin/Projects/github/writeapaper/kb_tree_matryoshka/experiments/msmarco

# Test JAX backend in baseline script
python -c "
from embedding_tools import EmbeddingStore
import numpy as np

store = EmbeddingStore(backend='jax', max_memory_gb=10.0)
embeddings = np.random.randn(1000, 1024).astype(np.float32)
store.add_embeddings(embeddings, dimension=1024)

query = np.random.randn(1024).astype(np.float32)
sims, indices = store.compute_similarity(query, dimension=1024, top_k=10)

print(f'✓ JAX backend works in kb_tree_matryoshka')
print(f'Top-10 indices: {indices[:10]}')
"
```

### 5.3 Performance Benchmark

Create quick benchmark script:

```python
# benchmark_jax.py
import time
import numpy as np
from embedding_tools import get_backend

backends = ['numpy', 'mlx', 'jax']
results = {}

for backend_name in backends:
    try:
        backend = get_backend(backend_name)

        # Create test data
        a = backend.create_array(np.random.randn(10000, 768).astype(np.float32))
        b = backend.create_array(np.random.randn(10000, 768).astype(np.float32))

        # Warm-up (for JIT compilation)
        _ = backend.cosine_similarity(a, b)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = backend.cosine_similarity(a, b)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000  # ms
        results[backend_name] = avg_time

        print(f"{backend_name:10s}: {avg_time:6.2f}ms")
    except ImportError:
        print(f"{backend_name:10s}: Not available")

# Print speedups
if 'numpy' in results:
    baseline = results['numpy']
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        print(f"{name} speedup vs NumPy: {speedup:.2f}x")
```

Run benchmark:
```bash
python benchmark_jax.py
```

Expected output:
```
numpy     :  85.23ms
mlx       :  12.45ms (6.8x faster)
jax       :   8.32ms (10.2x faster)
```

---

## Success Criteria

- [ ] JAX backend file created (`jax_backend.py`)
- [ ] All 17 abstract methods implemented
- [ ] JIT compilation working (speedup on repeated calls)
- [ ] Auto-detection includes JAX (MLX → JAX → PyTorch → NumPy)
- [ ] Tests pass (18+ tests in `test_jax_backend.py`)
- [ ] No regression in existing tests (52+ tests still passing)
- [ ] Documentation updated (README, JAX_GUIDE, DONE)
- [ ] Integration test with kb_tree_matryoshka successful
- [ ] Performance benchmark shows 5-10x speedup vs NumPy

---

## Estimated Timeline

| Phase | Estimated Time | Description |
|-------|----------------|-------------|
| Phase 1: Setup | 30 min | Install JAX, update pyproject.toml |
| Phase 2: Implementation | 3-4 hours | Create backend, update imports |
| Phase 3: Testing | 2-3 hours | Write and run tests |
| Phase 4: Documentation | 1 hour | Update docs, create guide |
| Phase 5: Validation | 30 min | Integration testing, benchmarks |
| **Total** | **6-8 hours** | **1-2 days of work** |

---

## Key Design Decisions

### 1. JIT Compilation Strategy
- **Pre-compile common operations** in `__init__` (cosine_similarity, normalize)
- Store compiled functions as instance attributes
- Trade-off: Longer initialization, faster subsequent calls

### 2. Random Number Generation
- JAX requires explicit PRNG keys (functional RNG)
- Use fixed seed (key=0) for `random_normal()` for reproducibility
- Can be extended to accept seed parameter if needed

### 3. Device Management
- JAX devices are objects, not strings
- Auto-detection: JAX puts best device first in `jax.devices()`
- Support 'gpu' and 'cpu' string aliases for consistency with other backends

### 4. NumPy Compatibility
- Use `jax.numpy` (jnp) for NumPy-compatible operations
- File I/O uses NumPy format (no native JAX serialization)
- Seamless conversion via `to_numpy()` and `from_numpy()`

---

## Future Enhancements (Optional)

### 1. Multi-Device Support
```python
# Shard embeddings across multiple GPUs
devices = jax.devices('gpu')
sharded_embeddings = jax.device_put_sharded(chunks, devices)
```

### 2. Advanced JIT Optimization
```python
# Static argument optimization for top_k parameter
@partial(jax.jit, static_argnums=(2,))
def similarity_search(query, corpus, top_k):
    sims = cosine_similarity(query, corpus)
    return jax.lax.top_k(sims, top_k)
```

### 3. TPU Support
- JAX has excellent TPU support
- Would enable cloud-scale experiments on Google Cloud TPUs
- Minimal code changes needed

---

## Notes

- JAX installation on Apple Silicon requires `jax-metal` package
- First run of JIT-compiled functions will be slower (compilation overhead)
- JAX uses functional programming paradigm (no in-place operations)
- Memory usage similar to MLX/PyTorch (unified memory on Mac)

**Next Step**: Review this plan, then proceed with Phase 1 when ready.
