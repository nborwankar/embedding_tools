"""Device and backend detection utilities.

Automatic detection of best available backend and device for the current platform.
"""

import platform
from typing import Dict, Optional, Tuple, Any


def detect_best_backend() -> str:
    """Detect best available backend for current platform.

    Detection priority:
    1. MLX (if on Mac and installed)
    2. PyTorch (if installed)
    3. NumPy (always available)

    Returns:
        Backend name ('mlx', 'torch', or 'numpy')
    """
    # Try MLX first (best on Mac)
    if platform.system() == 'Darwin':  # macOS
        try:
            import mlx.core
            return 'mlx'
        except ImportError:
            pass

    # Try PyTorch
    try:
        import torch
        return 'torch'
    except ImportError:
        pass

    # Fallback to NumPy
    return 'numpy'


def detect_best_device(backend: Optional[str] = None) -> Optional[str]:
    """Detect best available device for PyTorch backend.

    Only relevant for PyTorch backend. Returns None for other backends.

    Detection priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU

    Args:
        backend: Backend name. If None, detects backend first.

    Returns:
        Device name ('cuda', 'mps', 'cpu') or None if not PyTorch
    """
    if backend is None:
        backend = detect_best_backend()

    # Device detection only applies to PyTorch
    if backend != 'torch':
        return None

    try:
        import torch

        # Check CUDA first (NVIDIA GPUs)
        if torch.cuda.is_available():
            return 'cuda'

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            return 'mps'

        # Fallback to CPU
        return 'cpu'

    except ImportError:
        return None


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available devices and backends.

    Returns:
        Dictionary with device information:
        {
            'platform': 'Darwin' | 'Linux' | 'Windows',
            'backend_available': {
                'mlx': bool,
                'torch': bool,
                'numpy': bool
            },
            'torch_devices': {
                'cuda': bool,
                'mps': bool,
                'cpu': bool
            },
            'recommended_backend': 'mlx' | 'torch' | 'numpy',
            'recommended_device': 'cuda' | 'mps' | 'cpu' | None,
            'cuda_info': {...} | None,  # If CUDA available
            'system_info': {...}
        }
    """
    info = {
        'platform': platform.system(),
        'backend_available': {
            'mlx': False,
            'torch': False,
            'numpy': True  # Always available
        },
        'torch_devices': {
            'cuda': False,
            'mps': False,
            'cpu': True  # Always available
        },
        'recommended_backend': 'numpy',
        'recommended_device': None,
        'cuda_info': None,
        'system_info': {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }
    }

    # Check MLX availability
    try:
        import mlx.core
        info['backend_available']['mlx'] = True
        if platform.system() == 'Darwin':
            info['recommended_backend'] = 'mlx'
    except ImportError:
        pass

    # Check PyTorch and devices
    try:
        import torch
        info['backend_available']['torch'] = True

        # Check CUDA
        if torch.cuda.is_available():
            info['torch_devices']['cuda'] = True
            info['recommended_backend'] = 'torch'
            info['recommended_device'] = 'cuda'

            # Get CUDA info
            info['cuda_info'] = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'device_capability': torch.cuda.get_device_capability(0),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            }

        # Check MPS
        if torch.backends.mps.is_available():
            info['torch_devices']['mps'] = True
            if info['recommended_backend'] != 'mlx':  # MLX is faster on Mac
                info['recommended_backend'] = 'torch'
                info['recommended_device'] = 'mps'

        # If only CPU available
        if not info['torch_devices']['cuda'] and not info['torch_devices']['mps']:
            if info['recommended_backend'] == 'numpy':  # No MLX available
                info['recommended_backend'] = 'torch'
            info['recommended_device'] = 'cpu'

    except ImportError:
        pass

    return info


def detect_backend_with_fallback(prefer_performance: bool = True) -> Tuple[str, Optional[str]]:
    """Detect backend with fallback strategy.

    Args:
        prefer_performance: If True, prefer MLX over PyTorch MPS on Mac.
                          If False, prefer PyTorch for cross-platform consistency.

    Returns:
        Tuple of (backend_name, device)
        - backend_name: 'mlx', 'torch', or 'numpy'
        - device: 'cuda', 'mps', 'cpu', or None
    """
    system = platform.system()

    # Mac strategy
    if system == 'Darwin':
        if prefer_performance:
            # Performance priority: MLX > PyTorch MPS > PyTorch CPU > NumPy
            try:
                import mlx.core
                return ('mlx', None)
            except ImportError:
                pass

            try:
                import torch
                if torch.backends.mps.is_available():
                    return ('torch', 'mps')
                return ('torch', 'cpu')
            except ImportError:
                return ('numpy', None)
        else:
            # Cross-platform consistency: PyTorch > MLX > NumPy
            try:
                import torch
                if torch.backends.mps.is_available():
                    return ('torch', 'mps')
                return ('torch', 'cpu')
            except ImportError:
                try:
                    import mlx.core
                    return ('mlx', None)
                except ImportError:
                    return ('numpy', None)

    # Linux strategy
    elif system == 'Linux':
        # Prefer CUDA > CPU
        try:
            import torch
            if torch.cuda.is_available():
                return ('torch', 'cuda')
            return ('torch', 'cpu')
        except ImportError:
            return ('numpy', None)

    # Windows strategy
    elif system == 'Windows':
        # PyTorch with CUDA if available, else CPU
        try:
            import torch
            if torch.cuda.is_available():
                return ('torch', 'cuda')
            return ('torch', 'cpu')
        except ImportError:
            return ('numpy', None)

    # Unknown platform
    else:
        try:
            import torch
            return ('torch', 'cpu')
        except ImportError:
            return ('numpy', None)


def print_device_info():
    """Print human-readable device information."""
    info = get_device_info()

    print("=" * 70)
    print("Device Detection Summary")
    print("=" * 70)

    print(f"\nPlatform: {info['platform']}")
    print(f"Machine: {info['system_info']['machine']}")
    print(f"Processor: {info['system_info']['processor']}")

    print("\nAvailable Backends:")
    for backend, available in info['backend_available'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")

    print("\nPyTorch Devices:")
    for device, available in info['torch_devices'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {device}")

    if info['cuda_info']:
        print("\nCUDA Information:")
        print(f"  Device Count: {info['cuda_info']['device_count']}")
        print(f"  Device Name: {info['cuda_info']['device_name']}")
        print(f"  Compute Capability: {info['cuda_info']['device_capability']}")
        print(f"  Total Memory: {info['cuda_info']['total_memory_gb']:.1f} GB")

    print("\nRecommended Configuration:")
    print(f"  Backend: {info['recommended_backend']}")
    if info['recommended_device']:
        print(f"  Device: {info['recommended_device']}")

    print("=" * 70)


if __name__ == '__main__':
    # Run as standalone script
    print_device_info()

    print("\n" + "=" * 70)
    print("Detection Examples")
    print("=" * 70)

    print("\nBest backend:", detect_best_backend())
    print("Best device:", detect_best_device())

    backend, device = detect_backend_with_fallback(prefer_performance=True)
    print(f"\nWith performance priority: {backend}" + (f" (device={device})" if device else ""))

    backend, device = detect_backend_with_fallback(prefer_performance=False)
    print(f"With consistency priority: {backend}" + (f" (device={device})" if device else ""))
