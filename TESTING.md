# Testing Guide for embedding_tools

This document provides instructions for running tests in the embedding_tools repository.

## Prerequisites

Ensure you have the package installed in development mode:

```bash
cd embedding_tools
pip install -e ".[dev]"
```

To test specific backends, install their dependencies:

```bash
# For JAX backend tests
pip install -e ".[jax]"

# For PyTorch backend tests
pip install -e ".[torch]"

# For MLX backend tests (macOS only)
pip install -e ".[mlx]"

# For all backends
pip install -e ".[all]"
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

**JAX Backend Tests:**
```bash
pytest tests/test_jax_backend.py -v
```

**NumPy Backend Tests:**
```bash
pytest tests/test_arrays.py::TestNumpyBackend -v
```

**MLX Backend Tests (macOS only):**
```bash
pytest tests/test_arrays.py::TestMLXBackend -v
```

**PyTorch Backend Tests:**
```bash
pytest tests/test_torch_backend.py -v
```

**EmbeddingStore Tests:**
```bash
pytest tests/test_memory.py -v
```

**Configuration/Hashing Tests:**
```bash
pytest tests/test_config.py -v
```

**Installation Tests:**
```bash
pytest tests/test_installation.py -v
```

### Run Specific Test

```bash
pytest tests/test_jax_backend.py::TestJAXBackend::test_cosine_similarity -v
```

### Run with More Detail

**Show print statements and full output:**
```bash
pytest tests/test_jax_backend.py -v -s
```

**Show full tracebacks on failures:**
```bash
pytest tests/test_jax_backend.py -v --tb=long
```

**Stop on first failure:**
```bash
pytest tests/ -v -x
```

## Test Organization

The test suite is organized as follows:

```
tests/
├── test_arrays.py              # Backend operations (NumPy, MLX)
├── test_config.py              # Configuration hashing
├── test_installation.py        # Installation validation
├── test_jax_backend.py         # JAX backend (23 tests)
├── test_memory.py              # EmbeddingStore functionality
└── test_torch_backend.py       # PyTorch backend
```

## Expected Test Results

### On Linux (without MLX)

```
Total: 75 tests
- 71 PASSED
- 1 FAILED (test_mlx_backend_conditional - MLX not available)
- 3 ERROR (MLX tests - MLX not available on Linux)
```

### On macOS with MLX

```
Total: 75 tests
- 75 PASSED (all backends available)
```

### JAX Backend Tests Only

```
Total: 23 tests
- 23 PASSED
```

## Test Coverage

To run tests with coverage reporting:

```bash
pytest tests/ --cov=embedding_tools --cov-report=html
```

View the coverage report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests should pass on all platforms except:
- MLX tests will skip/fail on non-macOS platforms
- GPU-specific tests may skip if no GPU available

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure the package is installed:

```bash
pip install -e .
```

### Backend-Specific Failures

If a specific backend test fails:

1. Check if the backend is installed:
   ```bash
   python -c "import jax; print('JAX available')"
   python -c "import torch; print('PyTorch available')"
   python -c "import mlx.core; print('MLX available')"
   ```

2. Install the missing backend:
   ```bash
   pip install -e ".[jax]"    # or torch, mlx
   ```

### MLX on Linux

MLX is only available on Apple Silicon (M-series Macs). On Linux, MLX tests will error - this is expected behavior.

## Running Tests in Docker

```bash
# Build test container
docker build -t embedding-tools-test .

# Run tests
docker run embedding-tools-test pytest tests/ -v
```

## Quick Validation

After making changes, run this quick validation:

```bash
# Test core functionality
pytest tests/test_installation.py -v

# Test your specific backend
pytest tests/test_jax_backend.py -v

# Test no regressions
pytest tests/test_arrays.py::TestNumpyBackend -v
```

## Writing New Tests

When adding new functionality:

1. Add tests to the appropriate test file
2. Follow the existing test patterns
3. Use pytest fixtures for setup
4. Mark backend-specific tests with `@pytest.mark.skipif`

Example:

```python
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_jax_feature(self):
    backend = JAXBackend()
    # ... test code
```
