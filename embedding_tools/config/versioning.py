"""Configuration versioning using SHA-256 hashing."""

import hashlib
import json
from typing import Dict, Any


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        16-character hexadecimal hash string
    """
    # Convert config to deterministic JSON string
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_param_hash(**kwargs) -> str:
    """Compute hash from keyword arguments.

    Args:
        **kwargs: Parameters to hash

    Returns:
        16-character hexadecimal hash string

    Example:
        >>> hash_val = compute_param_hash(model='bert', dim=768, batch_size=32)
    """
    return compute_config_hash(kwargs)
