# Configuration-Driven Backend Selection

This directory contains examples of how to use configuration files to control backend and device selection without hardcoding platform-specific logic.

## Quick Start

### 1. Using Platform-Specific Configs

**Mac Development:**
```bash
python your_script.py --config config_mac.toml
```

**Linux Production:**
```bash
python your_script.py --config config_linux.toml
```

### 2. Using Cross-Platform Config with Auto-Detection

```bash
python your_script.py --config config_cross_platform.toml
```

This automatically detects the platform and selects the best backend.

### 3. Using Environment Variables

```bash
# Mac with MLX
export EMBEDDING_BACKEND=mlx
export EMBEDDING_MAX_MEMORY_GB=20.0

# Linux with CUDA
export EMBEDDING_BACKEND=torch
export EMBEDDING_DEVICE=cuda
export EMBEDDING_MAX_MEMORY_GB=40.0

# Run your application
python your_script.py
```

## Configuration Options

### Backend Types

| Type | Description | Platforms |
|------|-------------|-----------|
| `mlx` | Apple MLX (fastest on Mac) | macOS only |
| `torch` | PyTorch with device selection | All platforms |
| `numpy` | NumPy CPU (universal fallback) | All platforms |
| `auto` | Auto-detect best backend | All platforms |

### Device Options (PyTorch only)

| Device | Description | Requirements |
|--------|-------------|--------------|
| `cuda` | NVIDIA GPU | CUDA-capable GPU |
| `mps` | Apple Silicon GPU | M1/M2/M3 Mac |
| `cpu` | CPU fallback | Always available |
| `auto` | Auto-detect best device | Tries CUDA → MPS → CPU |

## Configuration File Examples

### Basic Config (config_mac.toml)

```toml
[experiment]
name = "mac_development"

[backend]
type = "mlx"  # Use MLX on Mac

[memory]
max_gb = 20.0

[model]
name = "all-mpnet-base-v2"
dimension = 768
```

### Cross-Platform Config (config_cross_platform.toml)

```toml
[experiment]
name = "cross_platform"

[backend]
type = "torch"
device = "auto"  # Auto-detect: cuda → mps → cpu

# Platform-specific overrides
[backend.platform_overrides.Darwin]
type = "mlx"  # Mac: prefer MLX

[backend.platform_overrides.Linux]
device = "cuda"  # Linux: prefer CUDA

# Fallback if primary fails
[backend.fallback]
enabled = true
strategy = "torch_cpu"  # Fallback to CPU

[memory]
max_gb = 20.0
```

### Environment Variables

```bash
# Required
EMBEDDING_BACKEND=torch          # 'mlx', 'torch', 'numpy', 'auto'
EMBEDDING_DEVICE=cuda            # 'cuda', 'mps', 'cpu', 'auto' (torch only)
EMBEDDING_MAX_MEMORY_GB=40.0

# Optional
DEPLOYMENT_PROFILE=production    # 'development', 'staging', 'production', 'ci'
```

## Usage in Your Code

### Option 1: Load from Config File

```python
from embedding_tools import EmbeddingStore
import toml

# Load config
config = toml.load('config_cross_platform.toml')

# Get backend settings
backend_type = config['backend']['type']
device = config['backend'].get('device')
max_memory = config['memory']['max_gb']

# Create store
store = EmbeddingStore(
    backend=backend_type if backend_type != 'auto' else None,
    max_memory_gb=max_memory,
    device=device
)
```

### Option 2: Use Environment Variables

```python
from embedding_tools import EmbeddingStore
import os

# Read from environment
backend = os.getenv('EMBEDDING_BACKEND', 'auto')
device = os.getenv('EMBEDDING_DEVICE', 'auto')
max_memory = float(os.getenv('EMBEDDING_MAX_MEMORY_GB', '8.0'))

# Create store
store = EmbeddingStore(
    backend=backend if backend != 'auto' else None,
    max_memory_gb=max_memory,
    device=device if device != 'auto' else None
)
```

### Option 3: Use Helper Function (Recommended)

```python
from config_driven_backend import create_embedding_store_from_config
import toml

# Load and apply config
config = toml.load('config_cross_platform.toml')
store = create_embedding_store_from_config(config)

# Store automatically handles:
# - Platform detection
# - Backend availability checking
# - Device fallback (cuda → mps → cpu)
# - Error handling
```

## Fallback Strategy

The configuration system includes robust fallback handling:

1. **Try primary backend/device** (from config)
2. **Check availability** (e.g., is CUDA actually present?)
3. **Try platform override** (if configured)
4. **Fallback to configured strategy**:
   - `mlx`: Best performance on Mac (20-40% faster than CPU)
   - `torch_cpu`: PyTorch with CPU (cross-platform, always works)
   - `numpy`: NumPy CPU (minimal dependencies, always works)
5. **Ultimate fallback**: NumPy CPU (always works)

### Fallback Strategy Options

| Strategy | Platform | Performance | Use Case |
|----------|----------|-------------|----------|
| `mlx` | Mac only | Fastest | Mac performance priority (GPU acceleration) |
| `torch_cpu` | All | Moderate | Cross-platform CPU fallback |
| `numpy` | All | Slowest | Minimal dependencies, always works |

**Example: Performance Priority on Mac**
```toml
[backend]
type = "torch"
device = "mps"

[backend.fallback]
enabled = true
strategy = "mlx"  # If MPS fails, use MLX (faster than CPU)
# If MLX fails, automatically falls back to torch_cpu
```

### Fallback Examples

**Config requests CUDA, but no GPU:**
```
Warning: Primary backend failed: CUDA requested but not available
Falling back to: torch_cpu
Created EmbeddingStore:
  Backend: TorchBackend
  Device: cpu
```

**Config requests MLX, but not installed:**
```
Warning: Primary backend failed: No module named 'mlx'
Falling back to: torch_cpu
Created EmbeddingStore:
  Backend: TorchBackend
  Device: cpu
```

**Config requests MPS, fallback to MLX (Mac performance priority):**
```
Warning: Primary backend failed: MPS requested but not available
Falling back to: mlx
Created EmbeddingStore:
  Backend: MLXBackend
  (MLX provides GPU acceleration - 20-40% faster than CPU)
```

**MLX fallback also fails (MLX not installed):**
```
Warning: Primary backend failed: CUDA requested but not available
Falling back to: mlx
Warning: MLX not available, falling back to torch_cpu
Created EmbeddingStore:
  Backend: TorchBackend
  Device: cpu
```

## Docker Deployment

### docker-compose.yml

```yaml
services:
  embedding-service:
    image: your-embedding-service
    environment:
      - EMBEDDING_BACKEND=torch
      - EMBEDDING_DEVICE=cuda
      - EMBEDDING_MAX_MEMORY_GB=40.0
    volumes:
      - ./config:/app/config
```

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install dependencies
RUN pip install embedding_tools

# Copy config
COPY config_linux.toml /app/config.toml

# Set environment variables
ENV EMBEDDING_BACKEND=torch
ENV EMBEDDING_DEVICE=cuda
ENV EMBEDDING_MAX_MEMORY_GB=40.0

CMD ["python", "app.py", "--config", "/app/config.toml"]
```

## Best Practices

### 1. Never Hardcode Platform Logic

❌ **Bad:**
```python
import platform
if platform.system() == 'Darwin':
    store = EmbeddingStore(backend='mlx', max_memory_gb=20.0)
else:
    store = EmbeddingStore(backend='torch', device='cuda', max_memory_gb=40.0)
```

✅ **Good:**
```python
config = load_config('config.toml')
store = create_embedding_store_from_config(config)
```

### 2. Use Platform Overrides in Config

```toml
[backend]
type = "torch"
device = "auto"

[backend.platform_overrides.Darwin]
type = "mlx"

[backend.platform_overrides.Linux]
device = "cuda"
```

### 3. Always Enable Fallback

```toml
[backend.fallback]
enabled = true
strategy = "torch_cpu"  # Safe fallback that always works
```

### 4. Use Environment Variables for Deployment

- **Development**: Use config files (config_mac.toml)
- **Production**: Use environment variables (docker-compose, k8s)
- **CI/CD**: Use environment variables with CPU fallback

### 5. Test Fallback Behavior

```python
# Simulate CUDA unavailable
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Config should fallback to CPU
config = load_config('config_linux.toml')
store = create_embedding_store_from_config(config)
assert store.backend.device == 'cpu'
```

## Deployment Profiles

### Development (Mac)
```toml
[backend]
type = "mlx"  # Fastest on Mac
[memory]
max_gb = 20.0
```

### Staging (Mac with PyTorch for consistency)
```toml
[backend]
type = "torch"
device = "mps"
[memory]
max_gb = 15.0
```

### Production (Linux with CUDA)
```toml
[backend]
type = "torch"
device = "cuda"
[memory]
max_gb = 40.0
```

### CI/CD (CPU only)
```toml
[backend]
type = "numpy"  # or torch with device=cpu
[memory]
max_gb = 4.0
```

## Running the Examples

```bash
# Run all examples
python config_driven_backend.py

# Individual examples
python -c "from config_driven_backend import example_json_config; example_json_config()"
python -c "from config_driven_backend import example_toml_config; example_toml_config()"
```

## Troubleshooting

**Problem**: CUDA requested but not available
**Solution**: Config will auto-fallback to CPU if `fallback.enabled = true`

**Problem**: MLX not installed
**Solution**: Config will fallback to PyTorch or NumPy

**Problem**: Different behavior on Mac vs Linux
**Solution**: Use `platform_overrides` to specify behavior per platform

**Problem**: Want to force CPU for testing
**Solution**: Set `EMBEDDING_DEVICE=cpu` or `device = "cpu"` in config
