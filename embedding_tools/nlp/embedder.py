"""Embedding wrapper with model registry and Matryoshka support.

Provides a single interface for NL and code embedding models.
Supports lazy loading, Matryoshka truncation, and legacy models
(kept in the registry so migration comparisons can load them).
"""

from typing import Optional, List

import numpy as np


class Embedder:
    """Config-driven, lazy-loading embedding wrapper.

    Usage:
        e = Embedder()                                # default: nomic-v1.5, 768D
        e = Embedder("jina-code-v2")                  # code model, 768D
        e = Embedder("nomic-v1.5", truncate_dim=256)  # Matryoshka at 256D

        embeddings = e.encode(["hello", "world"])     # (2, 768) or (2, 256)
        single = e.encode_single("hello")             # (768,) or (256,)
    """

    REGISTRY = {
        # ── Active models ──────────────────────────────────────────────
        "nomic-v1.5": {
            "hf_name": "nomic-ai/nomic-embed-text-v1.5",
            "dim": 768,
            "matryoshka": True,
            "matryoshka_dims": [768, 512, 256, 128, 64],
            "max_context": 8192,
            "trust_remote_code": True,
            "training_data": "235M pairs (search/cluster/STS/classification)",
            "normalize_before_truncate": True,
        },
        "jina-code-v2": {
            "hf_name": "jinaai/jina-embeddings-v2-base-code",
            "dim": 768,
            "matryoshka": False,
            "max_context": 8192,
            "trust_remote_code": False,
            "training_data": "150M+ code Q&A + docstring pairs, 30 languages",
        },
        # ── Legacy (kept for migration comparison) ──────────────────────
        "minilm-l6": {
            "hf_name": "all-MiniLM-L6-v2",
            "dim": 384,
            "matryoshka": False,
            "max_context": 384,
            "trust_remote_code": False,
            "training_data": "1B+ NLI/paraphrase pairs (distilled 6-layer)",
            "deprecated": True,
        },
        "mpnet": {
            "hf_name": "all-mpnet-base-v2",
            "dim": 768,
            "matryoshka": False,
            "max_context": 384,
            "trust_remote_code": False,
            "training_data": "1B+ NLI/paraphrase pairs (full 12-layer)",
            "deprecated": True,
        },
        "unixcoder": {
            "hf_name": "microsoft/unixcoder-base",
            "dim": 768,
            "matryoshka": False,
            "max_context": 512,
            "trust_remote_code": False,
            "training_data": "CodeSearchNet (6 languages)",
            "deprecated": True,
        },
    }

    def __init__(self, model_key: str = "nomic-v1.5", truncate_dim: Optional[int] = None):
        if model_key not in self.REGISTRY:
            raise KeyError(
                f"Unknown model key '{model_key}'. Available: {list(self.REGISTRY.keys())}"
            )
        self._model = None
        self._config = self.REGISTRY[model_key]
        self.model_key = model_key
        self.dim = truncate_dim or self._config["dim"]
        if truncate_dim and not self._config["matryoshka"]:
            raise ValueError(f"'{model_key}' does not support Matryoshka truncation")

    @property
    def full_dim(self) -> int:
        return self._config["dim"]

    @property
    def model_name(self) -> str:
        return self._config["hf_name"]

    @property
    def max_context(self) -> int:
        return self._config["max_context"]

    @property
    def is_deprecated(self) -> bool:
        return self._config.get("deprecated", False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._config["hf_name"],
                trust_remote_code=self._config.get("trust_remote_code", False),
            )

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings. Returns 2D float32 array (N x dim)."""
        self._load()
        embeddings = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        if self.dim < self._config["dim"]:
            if self._config.get("normalize_before_truncate"):
                import torch
                import torch.nn.functional as F

                t = torch.tensor(embeddings)
                t = F.layer_norm(t, normalized_shape=(t.shape[1],))
                embeddings = t[:, : self.dim].numpy()
            else:
                embeddings = embeddings[:, : self.dim]

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns 1D float32 array."""
        return self.encode([text])[0]
