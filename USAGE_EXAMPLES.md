# USAGE_EXAMPLES.md

Practical examples of using `embedding_tools` in typical experimental workflows, optimized for Apple Silicon with MLX acceleration.

## Table of Contents

1. [Basic Setup and Configuration](#1-basic-setup-and-configuration)
2. [Single Model Evaluation](#2-single-model-evaluation)
3. [A/B Testing Multiple Models](#3-ab-testing-multiple-models)
4. [Matryoshka Embeddings Workflow](#4-matryoshka-embeddings-workflow)
5. [Large-Scale Similarity Search](#5-large-scale-similarity-search)
6. [Experiment Versioning and Caching](#6-experiment-versioning-and-caching)
7. [Memory-Constrained Experiments](#7-memory-constrained-experiments)
8. [Cross-Platform Development](#8-cross-platform-development)

---

## 1. Basic Setup and Configuration

### Initial Setup with MLX (Apple Silicon)

```python
import numpy as np
from embedding_tools import get_backend, EmbeddingStore, compute_param_hash
from sentence_transformers import SentenceTransformer

# Auto-detect best backend (will use MLX on M-series Macs)
backend = get_backend()
print(f"Using backend: {backend.__class__.__name__}")  # MLXBackend on M2

# Or explicitly request MLX
backend = get_backend('mlx')

# Create embedding store with memory limit
store = EmbeddingStore(backend='mlx', max_memory_gb=20.0)

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"Model dimension: {model.get_sentence_embedding_dimension()}")  # 384
```

### Configuration Management

```python
# Define experiment configuration
config = {
    'model_name': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'chunk_size': 512,
    'chunk_overlap': 50,
    'batch_size': 32,
    'normalize': True
}

# Compute hash for versioning
exp_hash = compute_param_hash(**config)
print(f"Experiment ID: {exp_hash}")  # e.g., "a7b3c9d4e5f6g7h8"

# Use hash for file naming
output_dir = f"experiments/exp_{exp_hash}/"
embeddings_file = f"{output_dir}/embeddings.npz"
results_file = f"{output_dir}/results.json"
```

---

## 2. Single Model Evaluation

### End-to-End Evaluation Pipeline

```python
import numpy as np
from pathlib import Path
from embedding_tools import get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def evaluate_retrieval_model(
    documents: list[str],
    queries: list[str],
    relevance_judgments: dict,
    model_name: str = 'all-MiniLM-L6-v2'
):
    """Complete retrieval evaluation workflow."""

    # 1. Setup with MLX backend
    store = EmbeddingStore(backend='mlx', max_memory_gb=10.0)
    backend = store.backend

    # 2. Load model and encode documents
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True  # Get NumPy arrays
    ).astype(np.float32)

    # 3. Store embeddings (auto-converts to MLX internally)
    dimension = doc_embeddings.shape[1]
    store.add_embeddings(doc_embeddings, dimension=dimension)
    print(f"Stored {len(documents)} documents at {dimension}D")

    # 4. Evaluate queries
    results = []
    query_embeddings = model.encode(
        queries,
        batch_size=32,
        convert_to_numpy=True
    ).astype(np.float32)

    for i, (query, q_emb) in enumerate(zip(queries, query_embeddings)):
        # Similarity search using MLX acceleration
        similarities, indices = store.compute_similarity(
            q_emb,
            dimension=dimension,
            top_k=10
        )

        # Convert results to NumPy for processing
        top_indices = backend.to_numpy(indices)
        top_scores = backend.to_numpy(similarities)

        # Compute metrics
        relevant_docs = relevance_judgments.get(i, set())
        retrieved_docs = set(top_indices[:10])

        recall_at_10 = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0

        results.append({
            'query': query,
            'top_10_docs': top_indices.tolist(),
            'top_10_scores': top_scores.tolist(),
            'recall@10': recall_at_10
        })

    # 5. Aggregate metrics
    avg_recall = np.mean([r['recall@10'] for r in results])

    # 6. Memory usage report
    memory_info = store.get_memory_info()
    print(f"\nMemory usage: {memory_info['total_gb']:.2f} GB")
    print(f"Average Recall@10: {avg_recall:.3f}")

    return results, avg_recall

# Usage
documents = ["Document 1 text...", "Document 2 text...", ...]
queries = ["Query 1", "Query 2", ...]
relevance_judgments = {0: {0, 5, 12}, 1: {3, 8}, ...}  # query_id -> set of relevant doc_ids

results, recall = evaluate_retrieval_model(documents, queries, relevance_judgments)
```

---

## 3. A/B Testing Multiple Models

### Compare Multiple Embedding Models

```python
import json
import numpy as np
from pathlib import Path
from embedding_tools import get_backend, EmbeddingStore, compute_param_hash
from sentence_transformers import SentenceTransformer

def compare_models(
    documents: list[str],
    queries: list[str],
    relevance_judgments: dict,
    models: list[str]
):
    """A/B test multiple embedding models."""

    results_comparison = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # Create separate store for each model
        store = EmbeddingStore(backend='mlx', max_memory_gb=15.0)
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()

        # Configuration for this experiment
        config = {
            'model': model_name,
            'dimension': dimension,
            'corpus_size': len(documents)
        }
        exp_hash = compute_param_hash(**config)

        # Encode documents
        doc_embeddings = model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        store.add_embeddings(doc_embeddings, dimension=dimension)

        # Encode queries and evaluate
        query_embeddings = model.encode(
            queries,
            batch_size=64,
            convert_to_numpy=True
        ).astype(np.float32)

        recalls = []
        for i, q_emb in enumerate(query_embeddings):
            sims, indices = store.compute_similarity(
                q_emb,
                dimension=dimension,
                top_k=10
            )

            relevant_docs = relevance_judgments.get(i, set())
            retrieved_docs = set(store.backend.to_numpy(indices)[:10])
            recall = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0
            recalls.append(recall)

        # Store results
        results_comparison[model_name] = {
            'experiment_id': exp_hash,
            'dimension': dimension,
            'avg_recall@10': float(np.mean(recalls)),
            'std_recall@10': float(np.std(recalls)),
            'memory_gb': store.get_memory_info()['total_gb']
        }

        print(f"Dimension: {dimension}")
        print(f"Recall@10: {results_comparison[model_name]['avg_recall@10']:.3f}")
        print(f"Memory: {results_comparison[model_name]['memory_gb']:.2f} GB")

    # Save comparison
    with open('model_comparison.json', 'w') as f:
        json.dump(results_comparison, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, metrics in results_comparison.items():
        print(f"{model_name:40s} | Recall@10: {metrics['avg_recall@10']:.3f} | Dim: {metrics['dimension']:4d} | Mem: {metrics['memory_gb']:.1f}GB")

    return results_comparison

# Usage
models_to_test = [
    'sentence-transformers/all-MiniLM-L6-v2',      # 384D
    'sentence-transformers/all-mpnet-base-v2',     # 768D
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  # 384D
]

comparison = compare_models(documents, queries, relevance_judgments, models_to_test)
```

---

## 4. Matryoshka Embeddings Workflow

### Progressive Dimension Evaluation

```python
import numpy as np
from embedding_tools import get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def evaluate_matryoshka_dimensions(
    documents: list[str],
    queries: list[str],
    relevance_judgments: dict,
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    test_dimensions: list[int] = [64, 128, 256, 384, 512, 768]
):
    """Evaluate Matryoshka embeddings at multiple dimensions."""

    # Setup with MLX
    store = EmbeddingStore(backend='mlx', max_memory_gb=20.0)
    backend = store.backend

    # Encode at full dimension
    model = SentenceTransformer(model_name)
    full_dim = model.get_sentence_embedding_dimension()

    print(f"Encoding documents at full dimension: {full_dim}D")
    doc_embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Store full embeddings
    store.add_embeddings(doc_embeddings, dimension=full_dim)

    # Slice to all test dimensions
    for dim in test_dimensions:
        if dim < full_dim:
            sliced = store.slice_to_dimension(source_dim=full_dim, target_dim=dim)
            print(f"Sliced to {dim}D: {backend.get_shape(sliced)}")

    print(f"\nAvailable dimensions: {store.get_available_dimensions()}")

    # Encode queries at full dimension
    query_embeddings = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True
    ).astype(np.float32)

    # Evaluate at each dimension
    results_by_dimension = {}

    for dim in sorted(store.get_available_dimensions()):
        print(f"\nEvaluating at {dim}D...")

        recalls = []
        latencies = []

        for i, q_emb_full in enumerate(query_embeddings):
            # Truncate query to current dimension
            q_emb = q_emb_full[:dim]

            # Measure search time
            import time
            start = time.perf_counter()
            sims, indices = store.compute_similarity(
                q_emb,
                dimension=dim,
                top_k=10
            )
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Compute recall
            relevant_docs = relevance_judgments.get(i, set())
            retrieved_docs = set(backend.to_numpy(indices)[:10])
            recall = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0
            recalls.append(recall)

        # Store results
        memory_info = store.get_memory_info()
        dim_memory_mb = memory_info['dimensions'][dim]['memory_mb']

        results_by_dimension[dim] = {
            'recall@10': float(np.mean(recalls)),
            'avg_latency_ms': float(np.mean(latencies)),
            'memory_mb': float(dim_memory_mb),
            'compression_ratio': full_dim / dim
        }

        print(f"  Recall@10: {results_by_dimension[dim]['recall@10']:.3f}")
        print(f"  Latency: {results_by_dimension[dim]['avg_latency_ms']:.2f}ms")
        print(f"  Memory: {results_by_dimension[dim]['memory_mb']:.1f}MB")
        print(f"  Compression: {results_by_dimension[dim]['compression_ratio']:.1f}x")

    # Generate tradeoff analysis
    print(f"\n{'='*70}")
    print("DIMENSION TRADEOFF ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Dim':>6s} | {'Recall@10':>10s} | {'Latency(ms)':>12s} | {'Memory(MB)':>11s} | {'Compression':>11s}")
    print("-" * 70)

    for dim in sorted(results_by_dimension.keys()):
        r = results_by_dimension[dim]
        print(f"{dim:6d} | {r['recall@10']:10.3f} | {r['avg_latency_ms']:12.2f} | {r['memory_mb']:11.1f} | {r['compression_ratio']:11.1f}x")

    return results_by_dimension

# Usage
test_dims = [64, 128, 256, 384, 512, 768]
matryoshka_results = evaluate_matryoshka_dimensions(
    documents,
    queries,
    relevance_judgments,
    test_dimensions=test_dims
)
```

---

## 5. Large-Scale Similarity Search

### Efficient Nearest Neighbor Search

```python
import numpy as np
from embedding_tools import get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def large_scale_similarity_search(
    corpus_file: str,
    query_file: str,
    model_name: str = 'all-MiniLM-L6-v2',
    top_k: int = 100
):
    """Efficient similarity search over large corpus."""

    # Setup with MLX and large memory allocation
    store = EmbeddingStore(backend='mlx', max_memory_gb=50.0)
    backend = store.backend
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()

    # Load corpus in batches
    print("Loading and encoding corpus...")
    batch_size = 1000
    all_doc_embeddings = []

    with open(corpus_file, 'r') as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                embeddings = model.encode(
                    batch,
                    batch_size=64,
                    show_progress_bar=True,
                    convert_to_numpy=True
                ).astype(np.float32)
                all_doc_embeddings.append(embeddings)
                batch = []

        # Process remaining
        if batch:
            embeddings = model.encode(
                batch,
                batch_size=64,
                convert_to_numpy=True
            ).astype(np.float32)
            all_doc_embeddings.append(embeddings)

    # Concatenate and store
    doc_embeddings = np.vstack(all_doc_embeddings)
    print(f"Total documents: {len(doc_embeddings):,}")

    store.add_embeddings(doc_embeddings, dimension=dimension)

    # Memory report
    memory_info = store.get_memory_info()
    print(f"Memory usage: {memory_info['total_gb']:.2f} GB / {store.max_memory_bytes / 1e9:.2f} GB")

    # Load and process queries
    print("\nProcessing queries...")
    with open(query_file, 'r') as f:
        queries = [line.strip() for line in f]

    query_embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Batch similarity search
    all_results = []
    for i, q_emb in enumerate(query_embeddings):
        similarities, indices = store.compute_similarity(
            q_emb,
            dimension=dimension,
            top_k=top_k
        )

        # Convert to NumPy for storage
        all_results.append({
            'query_id': i,
            'indices': backend.to_numpy(indices).tolist(),
            'scores': backend.to_numpy(similarities).tolist()
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries")

    # Save results
    import json
    with open('search_results.json', 'w') as f:
        json.dump(all_results, f)

    print(f"\nCompleted {len(queries)} queries over {len(doc_embeddings):,} documents")
    print(f"Results saved to search_results.json")

    return all_results

# Usage
results = large_scale_similarity_search(
    corpus_file='corpus.txt',
    query_file='queries.txt',
    top_k=100
)
```

---

## 6. Experiment Versioning and Caching

### Automatic Cache Invalidation

```python
import json
import numpy as np
from pathlib import Path
from embedding_tools import compute_param_hash, get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def cached_embedding_experiment(
    documents: list[str],
    config: dict,
    cache_dir: str = 'cache/'
):
    """Run experiment with automatic caching and version detection."""

    # Compute experiment hash
    exp_hash = compute_param_hash(**config)
    cache_path = Path(cache_dir) / f"embeddings_{exp_hash}.npz"
    config_path = Path(cache_dir) / f"config_{exp_hash}.json"

    # Check if cached version exists
    if cache_path.exists() and config_path.exists():
        print(f"Loading cached embeddings for experiment {exp_hash}")

        # Verify config matches
        with open(config_path, 'r') as f:
            cached_config = json.load(f)

        cached_hash = compute_param_hash(**cached_config)
        if cached_hash == exp_hash:
            # Load from cache
            data = np.load(cache_path)
            embeddings = data['embeddings']
            print(f"Loaded {len(embeddings)} embeddings from cache")
            return embeddings, exp_hash, True  # True = from cache
        else:
            print("Config mismatch detected, recomputing...")

    # No cache or config mismatch - compute fresh
    print(f"Computing embeddings for experiment {exp_hash}")

    model = SentenceTransformer(config['model_name'])
    embeddings = model.encode(
        documents,
        batch_size=config.get('batch_size', 32),
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved embeddings to cache: {cache_path}")

    return embeddings, exp_hash, False  # False = freshly computed

# Example workflow with config evolution
configs = [
    {
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 32,
        'normalize': True
    },
    {
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 64,  # Changed batch size - same hash if model is same
        'normalize': True
    },
    {
        'model_name': 'all-mpnet-base-v2',  # Different model - new hash
        'batch_size': 32,
        'normalize': True
    }
]

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}")
    print(f"{'='*60}")

    embeddings, exp_hash, from_cache = cached_embedding_experiment(
        documents,
        config
    )

    print(f"Experiment ID: {exp_hash}")
    print(f"Cache hit: {from_cache}")
    print(f"Embeddings shape: {embeddings.shape}")
```

---

## 7. Memory-Constrained Experiments

### Working with Limited Memory

```python
import numpy as np
from embedding_tools import get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def memory_efficient_experiment(
    documents: list[str],
    queries: list[str],
    max_memory_gb: float = 5.0
):
    """Run experiments with strict memory constraints."""

    # Create store with memory limit
    store = EmbeddingStore(backend='mlx', max_memory_gb=max_memory_gb)
    backend = store.backend

    model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384

    # Strategy 1: Process in batches
    batch_size = 1000
    num_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"Processing {len(documents)} documents in {num_batches} batches")
    print(f"Memory limit: {max_memory_gb:.1f} GB\n")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]

        # Encode batch
        batch_embeddings = model.encode(
            batch_docs,
            batch_size=32,
            convert_to_numpy=True
        ).astype(np.float32)

        # Check memory before adding
        potential_memory = (
            store.get_total_memory_usage() +
            batch_embeddings.nbytes
        ) / 1e9

        if potential_memory > max_memory_gb:
            print(f"Batch {batch_idx}: Would exceed memory limit ({potential_memory:.2f} GB)")
            print("Switching to lower dimension...")

            # Use Matryoshka slicing to reduce memory
            if dimension > 128:
                # Slice existing embeddings to lower dimension
                if dimension in store.get_available_dimensions():
                    store.slice_to_dimension(source_dim=dimension, target_dim=128)
                    dimension = 128
                    batch_embeddings = batch_embeddings[:, :128]
                    print(f"Reduced to {dimension}D")

        try:
            store.add_embeddings(batch_embeddings, dimension=dimension)
            memory_info = store.get_memory_info()
            print(f"Batch {batch_idx}: Added {len(batch_embeddings)} docs, "
                  f"Memory: {memory_info['total_gb']:.2f} GB")
        except MemoryError as e:
            print(f"Batch {batch_idx}: MemoryError - {e}")
            break

    # Query with current dimension
    print(f"\nQuerying at {dimension}D...")
    query_embeddings = model.encode(
        queries,
        batch_size=32,
        convert_to_numpy=True
    ).astype(np.float32)

    # Truncate queries to match stored dimension
    query_embeddings = query_embeddings[:, :dimension]

    results = []
    for q_emb in query_embeddings:
        sims, indices = store.compute_similarity(
            q_emb,
            dimension=dimension,
            top_k=10
        )
        results.append({
            'indices': backend.to_numpy(indices).tolist(),
            'scores': backend.to_numpy(sims).tolist()
        })

    print(f"Completed {len(results)} queries")
    print(f"Final memory: {store.get_memory_info()['total_gb']:.2f} GB")

    return results

# Usage
results = memory_efficient_experiment(
    documents=large_corpus,  # e.g., 100K documents
    queries=test_queries,
    max_memory_gb=5.0  # Strict 5GB limit
)
```

---

## 8. Cross-Platform Development

### Mac Development → Linux Production

```python
import platform
import numpy as np
from embedding_tools import get_backend, EmbeddingStore
from sentence_transformers import SentenceTransformer

def cross_platform_experiment(documents: list[str], queries: list[str]):
    """Experiment that works on Mac (MLX) and Linux (PyTorch/CUDA)."""

    # Auto-detect best backend for platform
    if platform.system() == 'Darwin':  # macOS
        print("Running on macOS - using MLX backend")
        backend_name = 'mlx'
        device = None  # Not used for MLX
        max_memory_gb = 20.0  # M2 Max with 96GB RAM
    else:  # Linux or other
        print("Running on Linux - using PyTorch with CUDA")
        backend_name = 'torch'
        device = 'cuda'  # Use CUDA for GPU acceleration
        max_memory_gb = 10.0  # Typical server RAM

    backend = get_backend(backend_name, device=device)
    print(f"Backend: {backend.__class__.__name__}")
    if hasattr(backend, 'device'):
        print(f"Device: {backend.device}")

    # Create store with device specification
    store = EmbeddingStore(backend=backend_name, max_memory_gb=max_memory_gb, device=device)

    # Load model (works on all platforms)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384

    # Encode documents
    doc_embeddings = model.encode(
        documents,
        batch_size=64 if backend_name == 'mlx' else 32,  # Larger batches on MLX
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    store.add_embeddings(doc_embeddings, dimension=dimension)

    # Save embeddings in portable format (NumPy)
    # MLX backend automatically converts to NumPy for saving
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save_to_disk(tmpdir)
        print(f"Saved embeddings to {tmpdir} (NumPy format, portable)")

        # Can load on any platform
        new_store = EmbeddingStore(backend='numpy', max_memory_gb=10.0)
        new_store.load_from_disk(tmpdir)
        print("Loaded embeddings successfully (platform-independent)")

    # Query evaluation
    query_embeddings = model.encode(
        queries,
        batch_size=64 if backend_name == 'mlx' else 32,
        convert_to_numpy=True
    ).astype(np.float32)

    results = []
    for q_emb in query_embeddings:
        sims, indices = store.compute_similarity(
            q_emb,
            dimension=dimension,
            top_k=10
        )
        # Convert to NumPy for portable results
        results.append({
            'indices': backend.to_numpy(indices).tolist(),
            'scores': backend.to_numpy(sims).tolist()
        })

    print(f"\nProcessed {len(results)} queries")
    print(f"Platform: {platform.system()}")
    print(f"Backend: {backend_name}")

    return results

# Usage - same code works on Mac and Linux
results = cross_platform_experiment(documents, queries)
```

### Development-to-Production Pipeline

```python
import json
import platform
from pathlib import Path
from embedding_tools import compute_param_hash, get_backend, EmbeddingStore

def development_phase(documents: list[str], config: dict):
    """Development on Mac with MLX."""
    assert platform.system() == 'Darwin', "Development phase runs on macOS"

    # Use MLX for fast iteration
    store = EmbeddingStore(backend='mlx', max_memory_gb=20.0)

    # ... encode and experiment ...

    # Save for production
    exp_hash = compute_param_hash(**config)
    output_dir = Path(f"production_ready/exp_{exp_hash}")
    output_dir.mkdir(parents=True, exist_ok=True)

    store.save_to_disk(output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f)

    print(f"Saved production-ready artifacts to {output_dir}")
    return exp_hash

def production_phase(exp_hash: str):
    """Production on Linux with PyTorch/CUDA."""

    # Load from development artifacts
    input_dir = Path(f"production_ready/exp_{exp_hash}")

    # Use PyTorch with CUDA for GPU acceleration
    store = EmbeddingStore(backend='torch', max_memory_gb=50.0, device='cuda')
    store.load_from_disk(input_dir)

    with open(input_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Verify hash
    loaded_hash = compute_param_hash(**config)
    assert loaded_hash == exp_hash, "Config mismatch!"

    print(f"Loaded experiment {exp_hash} for production")
    print(f"Available dimensions: {store.get_available_dimensions()}")

    return store

# Workflow
if platform.system() == 'Darwin':
    config = {'model': 'all-MiniLM-L6-v2', 'dim': 384}
    exp_hash = development_phase(documents, config)
else:
    exp_hash = 'a7b3c9d4e5f6g7h8'  # From development
    store = production_phase(exp_hash)
```

---

### Explicit Device Configuration

```python
import torch
from embedding_tools import get_backend, EmbeddingStore

# Example 1: Explicit CUDA configuration for Linux production
if torch.cuda.is_available():
    backend = get_backend('torch', device='cuda')
    store = EmbeddingStore(backend='torch', max_memory_gb=40.0, device='cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Available CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Example 2: Explicit MPS configuration for Mac
elif torch.backends.mps.is_available():
    backend = get_backend('torch', device='mps')
    store = EmbeddingStore(backend='torch', max_memory_gb=20.0, device='mps')
    print("Using MPS (Metal Performance Shaders) on Apple Silicon")

# Example 3: CPU fallback
else:
    backend = get_backend('torch', device='cpu')
    store = EmbeddingStore(backend='torch', max_memory_gb=8.0, device='cpu')
    print("Using CPU (no GPU available)")

# Example 4: MLX for Apple Silicon (alternative to MPS)
try:
    backend = get_backend('mlx')
    store = EmbeddingStore(backend='mlx', max_memory_gb=20.0)
    print("Using MLX backend for Apple Silicon GPU")
except ImportError:
    print("MLX not available, falling back to PyTorch")

# Example 5: Configuration-driven device selection
config = {
    'backend': 'torch',
    'device': 'cuda',  # Options: 'cuda', 'mps', 'cpu'
    'max_memory_gb': 40.0
}

backend = get_backend(config['backend'], device=config['device'])
store = EmbeddingStore(
    backend=config['backend'],
    max_memory_gb=config['max_memory_gb'],
    device=config['device']
)
print(f"Backend: {backend.__class__.__name__}")
if hasattr(backend, 'device'):
    print(f"Device: {backend.device}")
```

---

## Summary

These examples demonstrate:

1. **GPU Acceleration**: CUDA (NVIDIA), MPS (Apple Silicon), or MLX (Apple Silicon)
2. **Device Configuration**: Explicit device specification for PyTorch backend
3. **Memory Management**: Configurable limits and monitoring
4. **Experiment Versioning**: SHA-256 hashing for reproducibility
5. **Matryoshka Support**: Progressive dimension evaluation
6. **Cross-Platform**: Portable artifacts between Mac and Linux
7. **Production-Ready**: Caching, batching, error handling

All workflows use the same clean API while optimizing for the underlying hardware.
