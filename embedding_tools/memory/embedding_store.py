"""In-memory embedding storage for experiments."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path

from embedding_tools.arrays import ArrayBackend, get_backend


class EmbeddingStore:
    """In-memory store for embeddings with multiple array backend support."""

    def __init__(self, backend: str = "numpy", max_memory_gb: float = 8.0, device: Optional[str] = None):
        """Initialize embedding store.

        Args:
            backend: Array backend ("numpy", "mlx", or "torch")
            max_memory_gb: Maximum memory usage in GB
            device: Device for PyTorch backend ('cuda', 'mps', 'cpu'). Auto-detects if None.
        """
        self.backend_name = backend
        self.backend: ArrayBackend = get_backend(backend, device=device)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)

        # Storage for embeddings by dimension
        self.embeddings: Dict[int, Any] = {}  # dimension -> array
        self.metadata: Dict[str, Any] = {}    # text_ids, labels, etc.
        self.dimension_info: Dict[int, Dict[str, Any]] = {}  # dimension -> info

    def add_embeddings(
        self,
        embeddings: Union[np.ndarray, Any],
        dimension: int,
        text_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add embeddings for a specific dimension.

        Args:
            embeddings: Embeddings array [n_samples, dimension]
            dimension: Embedding dimension
            text_ids: Text identifiers
            labels: Optional labels for classification
            metadata: Additional metadata
        """
        # Convert to backend format if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = self.backend.from_numpy(embeddings)

        # Check memory usage
        memory_usage = self.backend.get_memory_usage(embeddings)
        current_memory = self.get_total_memory_usage()

        if current_memory + memory_usage > self.max_memory_bytes:
            raise MemoryError(
                f"Adding embeddings would exceed memory limit "
                f"({(current_memory + memory_usage) / 1024**3:.2f}GB > "
                f"{self.max_memory_bytes / 1024**3:.2f}GB)"
            )

        # Store embeddings
        self.embeddings[dimension] = embeddings

        # Store metadata if provided
        if text_ids is not None:
            if 'text_ids' not in self.metadata:
                self.metadata['text_ids'] = text_ids
            elif self.metadata['text_ids'] != text_ids:
                print("Warning: text_ids mismatch with existing data")

        if labels is not None:
            if 'labels' not in self.metadata:
                self.metadata['labels'] = labels
            elif self.metadata['labels'] != labels:
                print("Warning: labels mismatch with existing data")

        # Store dimension info
        self.dimension_info[dimension] = {
            'shape': self.backend.get_shape(embeddings),
            'dtype': self.backend.get_dtype(embeddings),
            'memory_bytes': memory_usage,
            'metadata': metadata or {}
        }

    def get_embeddings(self, dimension: int) -> Optional[Any]:
        """Get embeddings for specific dimension."""
        return self.embeddings.get(dimension)

    def get_text_ids(self) -> Optional[List[str]]:
        """Get text identifiers."""
        return self.metadata.get('text_ids')

    def get_labels(self) -> Optional[List[str]]:
        """Get labels."""
        return self.metadata.get('labels')

    def slice_to_dimension(self, source_dim: int, target_dim: int) -> Optional[Any]:
        """Slice embeddings from larger dimension to smaller (Matryoshka)."""
        if source_dim not in self.embeddings:
            return None

        if target_dim > source_dim:
            raise ValueError(f"Target dimension {target_dim} > source dimension {source_dim}")

        source_embeddings = self.embeddings[source_dim]
        sliced_embeddings = self.backend.slice_last_dim(source_embeddings, target_dim)

        # Cache the sliced embeddings
        self.add_embeddings(
            sliced_embeddings,
            target_dim,
            metadata={'sliced_from': source_dim}
        )

        return sliced_embeddings

    def compute_similarity(
        self,
        query_emb: Union[np.ndarray, Any],
        dimension: int,
        top_k: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """Compute similarity between query and stored embeddings.

        Args:
            query_emb: Query embedding
            dimension: Dimension to use
            top_k: Return top-k most similar

        Returns:
            similarities, indices
        """
        if dimension not in self.embeddings:
            raise ValueError(f"No embeddings stored for dimension {dimension}")

        # Convert query to backend format if needed
        if isinstance(query_emb, np.ndarray):
            query_emb = self.backend.from_numpy(query_emb)

        # Ensure query has correct dimension
        if self.backend.get_shape(query_emb)[-1] != dimension:
            raise ValueError(f"Query dimension mismatch: got {self.backend.get_shape(query_emb)[-1]}, expected {dimension}")

        stored_embeddings = self.embeddings[dimension]
        similarities = self.backend.cosine_similarity(query_emb, stored_embeddings)

        if similarities.ndim == 2:
            similarities = similarities.reshape(-1)

        # Convert to numpy for sorting
        sim_np = self.backend.to_numpy(similarities)
        indices_np = np.argsort(sim_np)[::-1]

        if top_k is not None:
            indices_np = indices_np[:top_k].copy()  # Copy to avoid negative stride issues with PyTorch
            # Convert indices to backend format for indexing
            indices_backend = self.backend.create_array(indices_np, dtype='int64')
            similarities = similarities[indices_backend]

        # Return similarities in backend format, indices as numpy (standard for indices)
        return similarities, indices_np

    def get_available_dimensions(self) -> List[int]:
        """Get list of available dimensions."""
        return sorted(self.embeddings.keys())

    def get_total_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = 0
        for dim in self.embeddings:
            total += self.backend.get_memory_usage(self.embeddings[dim])
        return total

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        info = {
            'total_bytes': self.get_total_memory_usage(),
            'total_gb': self.get_total_memory_usage() / 1024**3,
            'max_gb': self.max_memory_bytes / 1024**3,
            'utilization': self.get_total_memory_usage() / self.max_memory_bytes,
            'dimensions': {}
        }

        for dim in self.embeddings:
            memory_bytes = self.backend.get_memory_usage(self.embeddings[dim])
            info['dimensions'][dim] = {
                'memory_bytes': memory_bytes,
                'memory_mb': memory_bytes / 1024**2,
                'shape': self.backend.get_shape(self.embeddings[dim]),
                'dtype': self.backend.get_dtype(self.embeddings[dim])
            }

        return info

    def save_to_disk(self, directory: Union[str, Path]) -> None:
        """Save all embeddings to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save embeddings for each dimension
        for dim in self.embeddings:
            filepath = directory / f"embeddings_{dim}d.npy"
            self.backend.save(self.embeddings[dim], str(filepath))

        # Save metadata
        import json
        with open(directory / "metadata.json", "w") as f:
            json.dump({
                'backend': self.backend_name,
                'dimensions': list(self.embeddings.keys()),
                'metadata': {k: v for k, v in self.metadata.items() if k != 'text_ids' and k != 'labels'},
                'dimension_info': {str(k): v for k, v in self.dimension_info.items()}
            }, f, indent=2)

        # Save text_ids and labels separately if they exist
        if 'text_ids' in self.metadata:
            with open(directory / "text_ids.txt", "w") as f:
                f.write("\n".join(self.metadata['text_ids']))

        if 'labels' in self.metadata:
            with open(directory / "labels.txt", "w") as f:
                f.write("\n".join(self.metadata['labels']))

    def load_from_disk(self, directory: Union[str, Path]) -> None:
        """Load embeddings from disk."""
        directory = Path(directory)

        # Load metadata
        import json
        with open(directory / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.backend_name = metadata['backend']
        self.backend = get_backend(self.backend_name)

        # Load embeddings
        for dim in metadata['dimensions']:
            filepath = directory / f"embeddings_{dim}d.npy"
            embeddings = self.backend.load(str(filepath))
            self.embeddings[dim] = embeddings

        # Load text_ids and labels if they exist
        text_ids_file = directory / "text_ids.txt"
        if text_ids_file.exists():
            with open(text_ids_file, "r") as f:
                self.metadata['text_ids'] = [line.strip() for line in f]

        labels_file = directory / "labels.txt"
        if labels_file.exists():
            with open(labels_file, "r") as f:
                self.metadata['labels'] = [line.strip() for line in f]

        # Restore dimension info
        self.dimension_info = {int(k): v for k, v in metadata['dimension_info'].items()}

    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        self.embeddings.clear()
        self.metadata.clear()
        self.dimension_info.clear()

    def __repr__(self) -> str:
        dims = self.get_available_dimensions()
        memory_gb = self.get_total_memory_usage() / 1024**3
        return f"EmbeddingStore(backend={self.backend_name}, dimensions={dims}, memory={memory_gb:.2f}GB)"