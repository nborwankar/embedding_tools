#!/usr/bin/env python
"""Configuration-driven backend selection for cross-platform deployment.

This example shows how to use config files (YAML/TOML/JSON) to control
backend and device selection without hardcoding platform-specific logic.
"""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from embedding_tools import get_backend, EmbeddingStore


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.

    Supports JSON, YAML, and TOML formats.
    """
    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path) as f:
            return json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.toml':
        import toml
        with open(config_path) as f:
            return toml.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def get_backend_from_config(config: Dict[str, Any]) -> tuple:
    """Get backend and device from configuration.

    Args:
        config: Configuration dictionary with 'backend' section

    Returns:
        (backend_name, device) tuple
    """
    backend_config = config.get('backend', {})

    # Option 1: Explicit backend and device in config
    backend_name = backend_config.get('type', None)  # 'numpy', 'mlx', 'torch'
    device = backend_config.get('device', None)      # 'cuda', 'mps', 'cpu'

    # Option 2: Platform-specific overrides
    platform_overrides = backend_config.get('platform_overrides', {})
    current_platform = platform.system()  # 'Darwin', 'Linux', 'Windows'

    if current_platform in platform_overrides:
        override = platform_overrides[current_platform]
        backend_name = override.get('type', backend_name)
        device = override.get('device', device)

    # Option 3: Auto-detection if not specified
    if backend_name is None:
        backend_name = 'auto'

    return backend_name if backend_name != 'auto' else None, device


def create_embedding_store_from_config(config: Dict[str, Any]) -> EmbeddingStore:
    """Create EmbeddingStore from configuration with robust fallback handling.

    Args:
        config: Configuration dictionary

    Returns:
        Configured EmbeddingStore instance
    """
    backend_name, device = get_backend_from_config(config)

    # Get memory configuration
    memory_config = config.get('memory', {})
    max_memory_gb = memory_config.get('max_gb', 8.0)

    # Get fallback configuration
    fallback_config = config.get('backend', {}).get('fallback', {})
    fallback_enabled = fallback_config.get('enabled', True)

    # Try to create store with primary configuration
    try:
        if backend_name:
            store = EmbeddingStore(
                backend=backend_name,
                max_memory_gb=max_memory_gb,
                device=device
            )
        else:
            # Auto-detection
            store = EmbeddingStore(max_memory_gb=max_memory_gb)

        # Verify device availability for PyTorch
        if backend_name == 'torch' and device:
            import torch
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")

    except (ImportError, RuntimeError) as e:
        print(f"Warning: Primary backend failed: {e}")

        if not fallback_enabled:
            raise

        # Fallback strategy
        fallback_strategy = fallback_config.get('strategy', 'torch_cpu')
        print(f"Falling back to: {fallback_strategy}")

        if fallback_strategy == 'mlx':
            # Try MLX first (best performance on Mac)
            try:
                store = EmbeddingStore(
                    backend='mlx',
                    max_memory_gb=max_memory_gb
                )
            except ImportError:
                print("Warning: MLX not available, falling back to torch_cpu")
                store = EmbeddingStore(
                    backend='torch',
                    max_memory_gb=max_memory_gb,
                    device='cpu'
                )
        elif fallback_strategy == 'torch_cpu':
            store = EmbeddingStore(
                backend='torch',
                max_memory_gb=max_memory_gb,
                device='cpu'
            )
        elif fallback_strategy == 'numpy':
            store = EmbeddingStore(
                backend='numpy',
                max_memory_gb=max_memory_gb
            )
        else:
            # Ultimate fallback: numpy
            print("Warning: Unknown fallback strategy, using numpy")
            store = EmbeddingStore(
                backend='numpy',
                max_memory_gb=max_memory_gb
            )

    # Print configuration
    print(f"Created EmbeddingStore:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print(f"  Max Memory: {max_memory_gb} GB")

    return store


def example_json_config():
    """Example 1: Using JSON configuration."""
    print("="*70)
    print("Example 1: JSON Configuration")
    print("="*70)

    config = {
        "backend": {
            "type": "torch",
            "device": "mps",  # Default
            "platform_overrides": {
                "Linux": {
                    "type": "torch",
                    "device": "cuda"
                },
                "Darwin": {
                    "type": "mlx"  # Use MLX on Mac for speed
                }
            }
        },
        "memory": {
            "max_gb": 20.0
        }
    }

    store = create_embedding_store_from_config(config)
    print()


def example_toml_config():
    """Example 2: Using TOML configuration (like embedding_expt)."""
    print("="*70)
    print("Example 2: TOML Configuration")
    print("="*70)

    # This mimics the embedding_expt/expt_config.toml pattern
    config = {
        "backend": {
            "type": "torch",
            "device": "auto",  # Auto-detect best device
        },
        "memory": {
            "max_gb": 40.0
        },
        "data": {
            "array_library": "mlx",  # Alternative config style
            "max_docs": 5000000
        }
    }

    store = create_embedding_store_from_config(config)
    print()


def example_environment_variables():
    """Example 3: Using environment variables."""
    print("="*70)
    print("Example 3: Environment Variables")
    print("="*70)

    # Read from environment with defaults
    backend_name = os.getenv('EMBEDDING_BACKEND', 'torch')
    device = os.getenv('EMBEDDING_DEVICE', 'auto')
    max_memory_gb = float(os.getenv('EMBEDDING_MAX_MEMORY_GB', '10.0'))

    if device == 'auto':
        device = None  # Let library auto-detect

    store = EmbeddingStore(
        backend=backend_name if backend_name != 'auto' else None,
        max_memory_gb=max_memory_gb,
        device=device
    )

    print(f"Created EmbeddingStore from environment:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print(f"  Max Memory: {max_memory_gb} GB")
    print()


def example_profile_based():
    """Example 4: Profile-based configuration."""
    print("="*70)
    print("Example 4: Profile-Based Configuration")
    print("="*70)

    # Define profiles for different deployment scenarios
    profiles = {
        "development": {
            "backend": {"type": "mlx"},  # Fastest on Mac
            "memory": {"max_gb": 20.0}
        },
        "staging": {
            "backend": {"type": "torch", "device": "mps"},
            "memory": {"max_gb": 15.0}
        },
        "production": {
            "backend": {"type": "torch", "device": "cuda"},
            "memory": {"max_gb": 40.0}
        },
        "ci": {
            "backend": {"type": "numpy"},  # CPU for CI/CD
            "memory": {"max_gb": 4.0}
        }
    }

    # Select profile from environment or default
    profile_name = os.getenv('DEPLOYMENT_PROFILE', 'development')
    config = profiles.get(profile_name, profiles['development'])

    print(f"Using profile: {profile_name}")
    store = create_embedding_store_from_config(config)
    print()


def example_conditional_config():
    """Example 5: Conditional configuration based on resources."""
    print("="*70)
    print("Example 5: Conditional Configuration (Resource-Based)")
    print("="*70)

    import torch

    # Detect available resources
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    try:
        import mlx.core
        has_mlx = True
    except ImportError:
        has_mlx = False

    # Build config based on what's available
    if has_mlx and platform.system() == 'Darwin':
        config = {
            "backend": {"type": "mlx"},
            "memory": {"max_gb": 20.0},
            "reason": "MLX available on Mac (fastest)"
        }
    elif has_cuda:
        config = {
            "backend": {"type": "torch", "device": "cuda"},
            "memory": {"max_gb": 40.0},
            "reason": "CUDA available (GPU acceleration)"
        }
    elif has_mps:
        config = {
            "backend": {"type": "torch", "device": "mps"},
            "memory": {"max_gb": 20.0},
            "reason": "MPS available (Apple Silicon GPU)"
        }
    else:
        config = {
            "backend": {"type": "numpy"},
            "memory": {"max_gb": 8.0},
            "reason": "CPU only (no GPU available)"
        }

    print(f"Auto-selected configuration: {config['reason']}")
    store = create_embedding_store_from_config(config)
    print()


def example_complete_workflow():
    """Example 6: Complete workflow with config file."""
    print("="*70)
    print("Example 6: Complete Workflow with Config File")
    print("="*70)

    # Create example config file
    config_file = Path("backend_config.json")

    example_config = {
        "experiment": {
            "name": "embedding_evaluation",
            "version": "1.0"
        },
        "backend": {
            "type": "torch",
            "device": "auto",
            "platform_overrides": {
                "Darwin": {"type": "mlx"},           # Mac: use MLX
                "Linux": {"device": "cuda"},         # Linux: use CUDA
                "Windows": {"device": "cpu"}         # Windows: use CPU
            }
        },
        "memory": {
            "max_gb": 20.0
        },
        "model": {
            "name": "all-MiniLM-L6-v2",
            "dimension": 384
        }
    }

    # Save config
    with open(config_file, 'w') as f:
        json.dump(example_config, f, indent=2)

    print(f"Created config file: {config_file}")
    print(f"Platform: {platform.system()}")

    # Load and use config
    config = load_config(config_file)
    store = create_embedding_store_from_config(config)

    # Use the store
    embeddings = np.random.randn(100, 384).astype(np.float32)
    store.add_embeddings(embeddings, dimension=384)
    print(f"Added {len(embeddings)} embeddings")

    # Cleanup
    config_file.unlink()
    print(f"Cleaned up config file")
    print()


def example_docker_deployment():
    """Example 7: Docker/Container deployment pattern."""
    print("="*70)
    print("Example 7: Docker/Container Deployment")
    print("="*70)

    # In Dockerfile or docker-compose.yml, set environment variables:
    # environment:
    #   - EMBEDDING_BACKEND=torch
    #   - EMBEDDING_DEVICE=cuda
    #   - EMBEDDING_MAX_MEMORY_GB=40.0

    # In your application, read from env vars
    backend = os.getenv('EMBEDDING_BACKEND', 'numpy')
    device = os.getenv('EMBEDDING_DEVICE', 'cpu')
    max_memory = float(os.getenv('EMBEDDING_MAX_MEMORY_GB', '8.0'))

    print("Docker environment configuration:")
    print(f"  EMBEDDING_BACKEND={backend}")
    print(f"  EMBEDDING_DEVICE={device}")
    print(f"  EMBEDDING_MAX_MEMORY_GB={max_memory}")

    store = EmbeddingStore(
        backend=backend,
        max_memory_gb=max_memory,
        device=device if backend == 'torch' else None
    )

    print(f"\nCreated store:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print()


if __name__ == '__main__':
    # Run all examples
    example_json_config()
    example_toml_config()
    example_environment_variables()
    example_profile_based()
    example_conditional_config()
    example_complete_workflow()
    example_docker_deployment()

    print("="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Use config files (JSON/TOML/YAML) for platform-specific settings")
    print("2. Use environment variables for deployment flexibility")
    print("3. Use profiles for different environments (dev/staging/prod)")
    print("4. Use platform_overrides for automatic Mac/Linux switching")
    print("5. Never hardcode backend/device choices in application code")
