"""Basic usage examples for embedding_tools."""

import numpy as np
from embedding_tools import get_backend, EmbeddingStore, compute_param_hash


def example_backends():
    """Example: Using different array backends."""
    print("="*60)
    print("Example 1: Array Backends")
    print("="*60)

    # Auto-detect backend
    backend = get_backend()
    print(f"Auto-detected backend: {backend.__class__.__name__}")

    # Create array
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    arr = backend.create_array(data)
    print(f"Created array shape: {backend.get_shape(arr)}")

    # Compute cosine similarity
    query = backend.create_array([1, 2, 3])
    sims = backend.cosine_similarity(query, arr)
    print(f"Similarities: {backend.to_numpy(sims)}")
    print()


def example_embedding_store():
    """Example: Using EmbeddingStore for memory-safe operations."""
    print("="*60)
    print("Example 2: EmbeddingStore")
    print("="*60)

    # Create store
    store = EmbeddingStore(backend='numpy', max_memory_gb=2.0)

    # Add embeddings at different dimensions
    emb_1024 = np.random.randn(1000, 1024).astype(np.float32)
    emb_512 = np.random.randn(1000, 512).astype(np.float32)

    store.add_embeddings(emb_1024, dimension=1024)
    store.add_embeddings(emb_512, dimension=512)

    print(f"Available dimensions: {store.get_available_dimensions()}")

    # Memory info
    info = store.get_memory_info()
    print(f"Total memory used: {info['total_gb']:.3f} GB")
    print(f"Memory by dimension:")
    for dim in info['dimensions']:
        mb = info['dimensions'][dim]['memory_mb']
        print(f"  {dim}D: {mb:.1f} MB")

    # Similarity search
    query = np.random.randn(1024).astype(np.float32)
    sims, indices = store.compute_similarity(query, dimension=1024, top_k=5)

    print(f"\nTop 5 similar indices: {store.backend.to_numpy(indices)}")
    print()


def example_matryoshka():
    """Example: Matryoshka embedding slicing."""
    print("="*60)
    print("Example 3: Matryoshka Slicing")
    print("="*60)

    store = EmbeddingStore(backend='numpy', max_memory_gb=1.0)

    # Add full-dimension embeddings
    emb_768 = np.random.randn(500, 768).astype(np.float32)
    store.add_embeddings(emb_768, dimension=768)
    print(f"Added 768D embeddings")

    # Slice to lower dimensions
    for target_dim in [384, 192, 96]:
        sliced = store.slice_to_dimension(source_dim=768, target_dim=target_dim)
        print(f"Sliced to {target_dim}D: shape = {store.backend.get_shape(sliced)}")

    print(f"\nNow have dimensions: {store.get_available_dimensions()}")
    print()


def example_versioning():
    """Example: Configuration versioning."""
    print("="*60)
    print("Example 4: Configuration Versioning")
    print("="*60)

    # Create experiment configuration
    config = {
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimension': 384,
        'batch_size': 32,
        'chunk_size': 512
    }

    # Compute hash
    exp_hash = compute_param_hash(**config)
    print(f"Experiment hash: {exp_hash}")

    # Use for caching
    cache_filename = f"embeddings_{exp_hash}.npz"
    print(f"Cache filename: {cache_filename}")

    # Same config = same hash
    exp_hash2 = compute_param_hash(**config)
    print(f"Same config gives same hash: {exp_hash == exp_hash2}")

    # Different config = different hash
    config2 = config.copy()
    config2['dimension'] = 768
    exp_hash3 = compute_param_hash(**config2)
    print(f"Different config gives different hash: {exp_hash != exp_hash3}")
    print()


def example_cross_backend():
    """Example: Converting between backends."""
    print("="*60)
    print("Example 5: Cross-Backend Conversion")
    print("="*60)

    # Create with NumPy
    np_backend = get_backend('numpy')
    arr_np = np_backend.create_array([[1, 2, 3], [4, 5, 6]])
    print(f"Created with NumPy: {type(arr_np)}")

    # Convert to standard NumPy
    arr_standard = np_backend.to_numpy(arr_np)
    print(f"Converted to NumPy: {type(arr_standard)}")

    # Could convert to MLX if available
    try:
        from embedding_tools import MLX_AVAILABLE
        if MLX_AVAILABLE:
            mlx_backend = get_backend('mlx')
            arr_mlx = mlx_backend.from_numpy(arr_standard)
            print(f"Converted to MLX: {type(arr_mlx)}")
            print("✓ MLX conversion works")
        else:
            print("⊘ MLX not available")
    except ImportError:
        print("⊘ MLX not installed")

    print()


if __name__ == '__main__':
    example_backends()
    example_embedding_store()
    example_matryoshka()
    example_versioning()
    example_cross_backend()

    print("="*60)
    print("All examples completed successfully!")
    print("="*60)
