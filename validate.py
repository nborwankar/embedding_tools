#!/usr/bin/env python
"""Quick validation script for embedding_tools installation.

Run this after pip install to verify everything works:
    python validate.py
"""

import sys


def main():
    print("="*70)
    print("embedding_tools Installation Validation")
    print("="*70)

    # Test imports
    print("\n[1/5] Testing package import...")
    try:
        import embedding_tools
        print(f"    ✓ embedding_tools {embedding_tools.__version__} imported successfully")
    except ImportError as e:
        print(f"    ✗ Failed to import embedding_tools: {e}")
        return 1

    # Test NumPy backend
    print("\n[2/5] Testing NumPy backend...")
    try:
        from embedding_tools import get_backend
        backend = get_backend('numpy')
        arr = backend.create_array([1, 2, 3])
        print(f"    ✓ NumPy backend works")
    except Exception as e:
        print(f"    ✗ NumPy backend failed: {e}")
        return 1

    # Test MLX backend
    print("\n[3/5] Testing MLX backend...")
    try:
        from embedding_tools import MLX_AVAILABLE
        if MLX_AVAILABLE:
            backend = get_backend('mlx')
            arr = backend.create_array([1, 2, 3])
            print(f"    ✓ MLX backend works")
        else:
            print(f"    ⊘ MLX not installed (optional)")
    except Exception as e:
        print(f"    ✗ MLX backend failed: {e}")
        print(f"    ⊘ MLX is optional, continuing...")

    # Test EmbeddingStore
    print("\n[4/5] Testing EmbeddingStore...")
    try:
        from embedding_tools import EmbeddingStore
        import numpy as np

        store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)
        embeddings = np.random.randn(100, 128).astype(np.float32)
        store.add_embeddings(embeddings, dimension=128)

        query = np.random.randn(128).astype(np.float32)
        sims, indices = store.compute_similarity(query, dimension=128, top_k=5)

        print(f"    ✓ EmbeddingStore works")
    except Exception as e:
        print(f"    ✗ EmbeddingStore failed: {e}")
        return 1

    # Test config versioning
    print("\n[5/5] Testing configuration versioning...")
    try:
        from embedding_tools import compute_config_hash, compute_param_hash

        hash1 = compute_config_hash({'model': 'bert', 'dim': 768})
        hash2 = compute_param_hash(model='bert', dim=768)

        if len(hash1) == 16 and len(hash2) == 16:
            print(f"    ✓ Configuration hashing works")
        else:
            print(f"    ✗ Hash length incorrect")
            return 1
    except Exception as e:
        print(f"    ✗ Configuration versioning failed: {e}")
        return 1

    # Summary
    print("\n" + "="*70)
    print("✓ All validation checks passed!")
    print("="*70)
    print("\nembedding_tools is ready to use. Try:")
    print("  from embedding_tools import get_backend, EmbeddingStore")
    print("  backend = get_backend()")
    print("\nRun full test suite with:")
    print("  pytest tests/ -v")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
