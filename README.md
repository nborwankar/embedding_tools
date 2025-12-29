# embedding_tools

[![PyPI version](https://badge.fury.io/py/embedding-tools.svg)](https://pypi.org/project/embedding-tools/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Utilities for embedding experiments with cross-platform array support**

`embedding_tools` provides a backend-agnostic interface for working with embeddings across NumPy, MLX (Apple Silicon), and PyTorch. It includes memory management, configuration versioning, and similarity search utilities optimized for machine learning research.

## Features

- 🔄 **Backend Abstraction**: Seamlessly switch between NumPy, MLX, and PyTorch
- 💾 **Memory Management**: Track and limit memory usage with `EmbeddingStore`
- 🔍 **Similarity Search**: Built-in cosine similarity and nearest neighbor search
- 📦 **Dimension Slicing**: Efficient truncation for Matryoshka embeddings
- 🔐 **Configuration Versioning**: SHA-256 hashing for reproducible experiments
- 🍎 **Apple Silicon Optimized**: Native MLX support for M-series Macs

## Installation

```bash
# Core (NumPy only)
pip install embedding_tools

# With MLX (Apple Silicon)
pip install embedding_tools[mlx]

# With PyTorch
pip install embedding_tools[torch]

# With JAX (GPU/TPU support)
pip install embedding_tools[jax]

# Everything
pip install embedding_tools[all]
```

### Development Installation

To contribute or use the latest development version:

```bash
git clone https://github.com/nborwankar/embedding_tools.git
cd embedding_tools
pip install -e ".[dev]"
```

## Quick Start

### Basic Array Operations

```python
from embedding_tools import get_backend

# Auto-detect best available backend
backend = get_backend()  # Uses MLX > JAX > PyTorch > NumPy

# Or specify explicitly
backend = get_backend('numpy')  # CPU
backend = get_backend('mlx')    # Apple Silicon GPU (fastest on Mac)
backend = get_backend('jax')    # GPU/TPU with JIT compilation
backend = get_backend('torch')  # PyTorch (CUDA/MPS/CPU)

# Create arrays
embeddings = backend.create_array([[1, 2, 3], [4, 5, 6]])

# Compute similarities
query = backend.create_array([1, 2, 3])
sims = backend.cosine_similarity(query, embeddings)

# Slice to lower dimensions (for Matryoshka embeddings)
truncated = backend.slice_last_dim(embeddings, dim=2)
```

### Memory-Safe Embedding Storage

```python
from embedding_tools import EmbeddingStore
import numpy as np

# Create store with memory limit
store = EmbeddingStore(backend='mlx', max_memory_gb=10.0)

# Add embeddings
embeddings_1024d = np.random.randn(10000, 1024).astype(np.float32)
store.add_embeddings(embeddings_1024d, dimension=1024)

# Slice to lower dimensions (Matryoshka)
embeddings_128d = store.slice_to_dimension(source_dim=1024, target_dim=128)

# Similarity search
query = np.random.randn(1024).astype(np.float32)
similarities, indices = store.compute_similarity(
    query,
    dimension=1024,
    top_k=10
)

# Check memory usage
info = store.get_memory_info()
print(f"Total memory: {info['total_gb']:.2f} GB")
```

### Configuration Versioning

```python
from embedding_tools import compute_config_hash, compute_param_hash

# Hash a configuration dict
config = {
    'model': 'sentence-transformers/all-MiniLM-L6-v2',
    'dimension': 384,
    'batch_size': 32
}
hash_val = compute_config_hash(config)  # Returns 16-char hex string

# Or use keyword arguments
hash_val = compute_param_hash(
    model='all-MiniLM-L6-v2',
    dimension=384,
    batch_size=32
)

# Use for automatic cache invalidation
cache_key = f"embeddings_{hash_val}.npz"
```

## Backend Comparison

| Backend | Hardware | Speed | Memory | Installation |
|---------|----------|-------|--------|--------------|
| NumPy   | CPU      | 1x    | System RAM | `pip install embedding_tools` |
| MLX     | Apple Silicon GPU | 3-5x | Unified memory | `pip install embedding_tools[mlx]` |
| JAX     | GPU/TPU (Metal/CUDA/ROCm) | 5-10x* | GPU VRAM | `pip install embedding_tools[jax]` |
| PyTorch | CUDA/MPS/CPU | 2-4x | GPU VRAM | `pip install embedding_tools[torch]` |

*Speed with JIT compilation on repeated operations

**Device Options for PyTorch:**
- `device='cuda'`: NVIDIA GPUs (Linux/Windows)
- `device='mps'`: Apple Silicon GPU (macOS)
- `device='cpu'`: CPU fallback (all platforms)

**Device Options for JAX:**
- `device='gpu'`: GPU acceleration (Metal/CUDA/ROCm)
- `device='cpu'`: CPU fallback
- `device=None`: Auto-detection (recommended)

```python
# Explicit device configuration
from embedding_tools import get_backend, EmbeddingStore

# PyTorch: CUDA for NVIDIA GPUs (Linux production)
backend = get_backend('torch', device='cuda')
store = EmbeddingStore(backend='torch', max_memory_gb=40.0, device='cuda')

# PyTorch: MPS for Apple Silicon
backend = get_backend('torch', device='mps')
store = EmbeddingStore(backend='torch', max_memory_gb=20.0, device='mps')

# JAX: GPU acceleration (auto-detects Metal/CUDA/ROCm)
backend = get_backend('jax', device='gpu')
store = EmbeddingStore(backend='jax', max_memory_gb=20.0, device='gpu')

# Auto-detection (recommended)
backend = get_backend('torch')  # Automatically picks best device
backend = get_backend('jax')    # Automatically picks best device
```

## Installation Validation

Run validation tests after installation:

```bash
pytest tests/test_installation.py -v
```

Or run directly:

```bash
python tests/test_installation.py
```

Expected output:
```
============================================================
embedding_tools Installation Validation Summary
============================================================
Version: 0.1.0
NumPy backend: ✓ Available
MLX backend: ✓ Available
Auto-detected backend: MLXBackend

All core functionality tests passed!
============================================================
```

## Development

```bash
# Clone repository
git clone https://github.com/nborwankar/embedding_tools.git
cd embedding_tools

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black .
isort .

# Lint
flake8 embedding_tools/
```

## API Reference

### Array Backends

#### `get_backend(backend_name=None, device=None)`

Get array backend instance.

**Parameters:**
- `backend_name` (str, optional): 'numpy', 'mlx', 'jax', or 'torch'. Auto-detects if None.
- `device` (str, optional): Device specification for JAX/PyTorch backends. Auto-detects if None.

**Returns:** ArrayBackend instance

#### `ArrayBackend` Methods

- `create_array(data, dtype=None)` - Create array from data
- `zeros(shape, dtype=None)` - Create zero-filled array
- `ones(shape, dtype=None)` - Create one-filled array
- `random_normal(shape, mean=0.0, std=1.0)` - Random normal array
- `dot(a, b)` - Dot product
- `cosine_similarity(a, b)` - Cosine similarity matrix
- `normalize(a, axis=-1)` - L2 normalization
- `concatenate(arrays, axis=0)` - Concatenate arrays
- `stack(arrays, axis=0)` - Stack arrays
- `slice_last_dim(array, dim)` - Slice to dimension
- `to_numpy(array)` - Convert to NumPy
- `from_numpy(array)` - Convert from NumPy
- `save(array, filepath)` - Save to file
- `load(filepath)` - Load from file
- `get_memory_usage(array)` - Memory in bytes
- `get_shape(array)` - Array shape
- `get_dtype(array)` - Array dtype

### Memory Management

#### `EmbeddingStore(backend='numpy', max_memory_gb=8.0)`

In-memory embedding storage with memory limits.

**Methods:**
- `add_embeddings(embeddings, dimension, text_ids=None, labels=None, metadata=None)`
- `get_embeddings(dimension)` - Retrieve embeddings
- `slice_to_dimension(source_dim, target_dim)` - Matryoshka slicing
- `compute_similarity(query_emb, dimension, top_k=None)` - Similarity search
- `get_available_dimensions()` - List stored dimensions
- `get_total_memory_usage()` - Total memory in bytes
- `get_memory_info()` - Detailed memory statistics
- `save_to_disk(directory)` - Save all embeddings
- `load_from_disk(directory)` - Load all embeddings

### Configuration

#### `compute_config_hash(config)`

Compute SHA-256 hash of configuration dictionary.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:** 16-character hex string

#### `compute_param_hash(**kwargs)`

Convenience function for hashing keyword arguments.

**Returns:** 16-character hex string

## Use Cases

### Matryoshka Embeddings

```python
from embedding_tools import EmbeddingStore, get_backend

backend = get_backend('mlx')
store = EmbeddingStore(backend='mlx', max_memory_gb=20)

# Train model to produce 1024D embeddings
full_embeddings = model.encode(documents)  # (N, 1024)
store.add_embeddings(full_embeddings, dimension=1024)

# Get truncated versions for different use cases
embeddings_512 = store.slice_to_dimension(1024, 512)  # Moderate accuracy
embeddings_128 = store.slice_to_dimension(1024, 128)  # Fast search
embeddings_32 = store.slice_to_dimension(1024, 32)    # Ultra-fast

# Compare at different dimensions
for dim in [32, 128, 512, 1024]:
    sims, indices = store.compute_similarity(query, dim, top_k=10)
    print(f"{dim}D recall@10: {compute_recall(indices, ground_truth)}")
```

### Cross-Platform Development

```python
from embedding_tools import get_backend

# Development on Mac (uses MLX for speed)
if platform.system() == 'Darwin':
    backend = get_backend('mlx')
# Production on Linux (uses NumPy or CUDA)
else:
    backend = get_backend('numpy')

# Same code works everywhere
embeddings = backend.create_array(data)
similarities = backend.cosine_similarity(query, embeddings)
```

### Experiment Versioning

```python
from embedding_tools import compute_param_hash
import os

# Compute hash of experiment parameters
exp_hash = compute_param_hash(
    model='all-MiniLM-L6-v2',
    chunk_size=512,
    overlap=50,
    dimension=384
)

# Check if results exist
results_file = f'results_{exp_hash}.json'
if os.path.exists(results_file):
    print("Loading cached results...")
    results = load_results(results_file)
else:
    print("Running new experiment...")
    results = run_experiment()
    save_results(results, results_file)
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Citation

If you use embedding_tools in your research, please cite:

```bibtex
@software{embedding_tools2025,
  title = {embedding_tools: Utilities for embedding experiments},
  author = {Nitin Borwankar},
  year = {2025},
  url = {https://github.com/nborwankar/embedding_tools}
}
```
