#!/usr/bin/env python
"""Device detection workflow examples.

This shows how to use the device detection utilities to automatically
configure backends without config files or environment variables.
"""

import numpy as np
from embedding_tools import (
    detect_best_backend,
    detect_best_device,
    get_device_info,
    get_backend,
    EmbeddingStore
)


def example_simple_detection():
    """Example 1: Simple automatic detection."""
    print("=" * 70)
    print("Example 1: Simple Automatic Detection")
    print("=" * 70)

    # Detect best backend
    backend_name = detect_best_backend()
    print(f"Best backend: {backend_name}")

    # Detect best device (for PyTorch)
    device = detect_best_device(backend_name)
    print(f"Best device: {device if device else 'N/A (not PyTorch)'}")

    # Create store with detected configuration
    if device:
        store = EmbeddingStore(backend=backend_name, max_memory_gb=10.0, device=device)
    else:
        store = EmbeddingStore(backend=backend_name, max_memory_gb=10.0)

    print(f"\nCreated store:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print()


def example_detailed_info():
    """Example 2: Get detailed device information."""
    print("=" * 70)
    print("Example 2: Detailed Device Information")
    print("=" * 70)

    info = get_device_info()

    print(f"\nPlatform: {info['platform']}")
    print(f"Machine: {info['system_info']['machine']}")

    print("\nAvailable Backends:")
    for backend, available in info['backend_available'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")

    print("\nPyTorch Devices:")
    for device, available in info['torch_devices'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {device}")

    if info['cuda_info']:
        print("\nCUDA Details:")
        print(f"  GPU: {info['cuda_info']['device_name']}")
        print(f"  Memory: {info['cuda_info']['total_memory_gb']:.1f} GB")

    print(f"\nRecommended: {info['recommended_backend']}", end="")
    if info['recommended_device']:
        print(f" (device={info['recommended_device']})")
    else:
        print()
    print()


def example_auto_configure():
    """Example 3: Fully automatic configuration."""
    print("=" * 70)
    print("Example 3: Fully Automatic Configuration")
    print("=" * 70)

    # Get device info
    info = get_device_info()

    # Create store with recommended settings
    backend = info['recommended_backend']
    device = info['recommended_device']

    if device:
        store = EmbeddingStore(backend=backend, max_memory_gb=20.0, device=device)
    else:
        store = EmbeddingStore(backend=backend, max_memory_gb=20.0)

    print(f"Auto-configured store:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")

    # Use the store
    embeddings = np.random.randn(100, 384).astype(np.float32)
    store.add_embeddings(embeddings, dimension=384)
    print(f"  Stored: {len(embeddings)} embeddings")
    print()


def example_performance_priority():
    """Example 4: Choose backend based on performance priority."""
    print("=" * 70)
    print("Example 4: Performance Priority Selection")
    print("=" * 70)

    from embedding_tools.utils.device_detection import detect_backend_with_fallback

    # Performance priority (prefer MLX on Mac)
    backend, device = detect_backend_with_fallback(prefer_performance=True)
    print(f"Performance priority: {backend}", end="")
    if device:
        print(f" (device={device})")
    else:
        print()

    # Consistency priority (prefer PyTorch cross-platform)
    backend, device = detect_backend_with_fallback(prefer_performance=False)
    print(f"Consistency priority: {backend}", end="")
    if device:
        print(f" (device={device})")
    else:
        print()
    print()


def example_conditional_workflow():
    """Example 5: Conditional workflow based on available hardware."""
    print("=" * 70)
    print("Example 5: Conditional Workflow Based on Hardware")
    print("=" * 70)

    info = get_device_info()

    # Adjust batch size based on available hardware
    if info['cuda_info']:
        # CUDA available - use large batches
        batch_size = 128
        max_memory = info['cuda_info']['total_memory_gb'] * 0.8  # Use 80% of GPU memory
        print(f"CUDA detected: Using large batches (batch_size={batch_size})")
        print(f"GPU Memory: {max_memory:.1f} GB available")

    elif info['torch_devices']['mps']:
        # MPS available - moderate batches
        batch_size = 64
        max_memory = 20.0  # Conservative for unified memory
        print(f"MPS detected: Using moderate batches (batch_size={batch_size})")
        print(f"Unified Memory: {max_memory:.1f} GB allocated")

    elif info['backend_available']['mlx']:
        # MLX available - moderate batches
        batch_size = 64
        max_memory = 20.0
        print(f"MLX detected: Using moderate batches (batch_size={batch_size})")
        print(f"Unified Memory: {max_memory:.1f} GB allocated")

    else:
        # CPU only - small batches
        batch_size = 16
        max_memory = 8.0
        print(f"CPU only: Using small batches (batch_size={batch_size})")
        print(f"System Memory: {max_memory:.1f} GB allocated")

    # Create store with adaptive settings
    backend = info['recommended_backend']
    device = info['recommended_device']

    if device:
        store = EmbeddingStore(backend=backend, max_memory_gb=max_memory, device=device)
    else:
        store = EmbeddingStore(backend=backend, max_memory_gb=max_memory)

    print(f"\nConfigured store:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max memory: {max_memory:.1f} GB")
    print()


def example_print_summary():
    """Example 6: Print complete device summary."""
    print("=" * 70)
    print("Example 6: Complete Device Summary")
    print("=" * 70)

    from embedding_tools.utils.device_detection import print_device_info
    print_device_info()


def example_one_liner():
    """Example 7: One-liner automatic configuration."""
    print("=" * 70)
    print("Example 7: One-Liner Automatic Configuration")
    print("=" * 70)

    # Create store with automatic detection (one line!)
    store = EmbeddingStore(
        backend=detect_best_backend(),
        max_memory_gb=20.0,
        device=detect_best_device()
    )

    print(f"Created with one-liner:")
    print(f"  Backend: {store.backend.__class__.__name__}")
    if hasattr(store.backend, 'device'):
        print(f"  Device: {store.backend.device}")
    print()


def example_fallback_chain():
    """Example 8: Show complete fallback chain."""
    print("=" * 70)
    print("Example 8: Fallback Chain Visualization")
    print("=" * 70)

    info = get_device_info()

    print("Available options (in priority order):")
    priority = []

    # MLX (Mac only)
    if info['backend_available']['mlx']:
        priority.append("✓ MLX (fastest on Mac)")
    else:
        priority.append("✗ MLX (not available)")

    # PyTorch CUDA
    if info['torch_devices']['cuda']:
        priority.append("✓ PyTorch + CUDA (fastest on Linux)")
    else:
        priority.append("✗ PyTorch + CUDA (not available)")

    # PyTorch MPS
    if info['torch_devices']['mps']:
        priority.append("✓ PyTorch + MPS (good on Mac)")
    else:
        priority.append("✗ PyTorch + MPS (not available)")

    # PyTorch CPU
    if info['backend_available']['torch']:
        priority.append("✓ PyTorch + CPU (cross-platform)")
    else:
        priority.append("✗ PyTorch + CPU (not available)")

    # NumPy
    priority.append("✓ NumPy (always available)")

    for i, option in enumerate(priority, 1):
        print(f"{i}. {option}")

    print(f"\nSelected: {info['recommended_backend']}", end="")
    if info['recommended_device']:
        print(f" + {info['recommended_device']}")
    else:
        print()
    print()


if __name__ == '__main__':
    example_simple_detection()
    example_detailed_info()
    example_auto_configure()
    example_performance_priority()
    example_conditional_workflow()
    example_print_summary()
    example_one_liner()
    example_fallback_chain()

    print("=" * 70)
    print("All device detection examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Use detect_best_backend() for automatic backend selection")
    print("2. Use detect_best_device() for automatic device selection (PyTorch)")
    print("3. Use get_device_info() for detailed hardware information")
    print("4. Adjust batch sizes and memory based on available hardware")
    print("5. No config files or environment variables needed!")
