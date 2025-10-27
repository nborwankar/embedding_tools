#!/usr/bin/env python
"""Quick test of PyTorch backend implementation."""

import sys
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed, skipping test")
    sys.exit(0)

from embedding_tools import get_backend, EmbeddingStore

def test_torch_backend():
    """Test PyTorch backend with device detection."""

    print("="*60)
    print("Testing PyTorch Backend")
    print("="*60)

    # Test 1: Auto-detection
    print("\n[1] Auto-detection:")
    backend = get_backend('torch')
    print(f"  Backend: {backend.__class__.__name__}")
    print(f"  Device: {backend.device}")

    # Test 2: Explicit device
    if torch.backends.mps.is_available():
        print("\n[2] Explicit MPS:")
        backend_mps = get_backend('torch', device='mps')
        print(f"  Device: {backend_mps.device}")
    elif torch.cuda.is_available():
        print("\n[2] Explicit CUDA:")
        backend_cuda = get_backend('torch', device='cuda')
        print(f"  Device: {backend_cuda.device}")
    else:
        print("\n[2] CPU only:")
        backend_cpu = get_backend('torch', device='cpu')
        print(f"  Device: {backend_cpu.device}")

    # Test 3: Basic operations
    print("\n[3] Basic operations:")
    data = [[1, 2, 3], [4, 5, 6]]
    arr = backend.create_array(data)
    print(f"  Created array: {backend.get_shape(arr)}")
    print(f"  Dtype: {backend.get_dtype(arr)}")

    # Test 4: Cosine similarity
    print("\n[4] Cosine similarity:")
    query = backend.create_array([1, 2, 3])
    sims = backend.cosine_similarity(query, arr)
    print(f"  Similarities: {backend.to_numpy(sims)}")

    # Test 5: Dimension slicing
    print("\n[5] Dimension slicing:")
    emb = backend.create_array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    sliced = backend.slice_last_dim(emb, 3)
    print(f"  Original shape: {backend.get_shape(emb)}")
    print(f"  Sliced shape: {backend.get_shape(sliced)}")

    # Test 6: EmbeddingStore integration
    print("\n[6] EmbeddingStore integration:")
    device_str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    store = EmbeddingStore(backend='torch', max_memory_gb=1.0, device=device_str)

    embeddings = np.random.randn(100, 128).astype(np.float32)
    store.add_embeddings(embeddings, dimension=128)
    print(f"  Stored {len(embeddings)} embeddings at 128D")

    query = np.random.randn(128).astype(np.float32)
    sims, indices = store.compute_similarity(query, dimension=128, top_k=5)
    print(f"  Top 5 indices: {indices}")  # indices is already numpy

    # Test 7: Memory info
    print("\n[7] Memory info:")
    memory_info = store.get_memory_info()
    print(f"  Total memory: {memory_info['total_gb']:.3f} GB")
    print(f"  128D memory: {memory_info['dimensions'][128]['memory_mb']:.1f} MB")

    print("\n" + "="*60)
    print("✓ All PyTorch backend tests passed!")
    print("="*60)

    return 0

if __name__ == '__main__':
    sys.exit(test_torch_backend())
