# Experiment Versioning Best Practices

**Last Updated**: 2025-10-26

---

## The Problem

The `compute_param_hash()` function in `embedding_tools` **only hashes configuration parameters**, not code implementation.

### What This Means

**Config changes are detected** ✅:
```python
# Different configs = different hashes
config1 = {'model': 'all-MiniLM', 'dim': 384}  # hash: "abc123"
config2 = {'model': 'all-MiniLM', 'dim': 512}  # hash: "def456" (different!)
```

**Code changes are NOT detected** ❌:
```python
# VERSION 1 (buggy)
def encode_documents(docs, model):
    embeddings = model.encode(docs)
    return embeddings  # Bug: not normalized!

config = {'model': 'all-MiniLM', 'dim': 384}
hash_v1 = compute_param_hash(**config)  # "abc123"

# VERSION 2 (fixed)
def encode_documents(docs, model):
    embeddings = model.encode(docs)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Fixed!

config = {'model': 'all-MiniLM', 'dim': 384}  # SAME CONFIG
hash_v2 = compute_param_hash(**config)  # "abc123" - SAME HASH! ❌
```

**Result**: Cached results from buggy version might be reused with fixed code, causing incorrect results.

---

## Solutions

### Solution 1: Add Code Version to Config ✅ **RECOMMENDED**

Add a `pipeline_version` parameter that you manually increment when code changes:

```python
from embedding_tools import compute_param_hash

config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'batch_size': 32,
    'normalize': True,

    # Track code/implementation version
    'pipeline_version': '2.0',  # Increment when code changes
    'implementation_notes': 'Fixed L2 normalization bug'  # Optional documentation
}

exp_hash = compute_param_hash(**config)  # Now reflects code version
```

**Version Bump Rules**:
| Change Type | Version Change | Example |
|-------------|----------------|---------|
| Bug fix in code | Minor bump | `2.0` → `2.1` |
| Algorithm change | Major bump | `2.0` → `3.0` |
| Config-only change | No bump | `2.0` → `2.0` (hash changes automatically) |
| Documentation only | No bump | `2.0` → `2.0` (doesn't affect results) |

---

### Solution 2: Include Git Commit Hash

Automatically track code changes via git commit hash:

```python
import subprocess
from embedding_tools import compute_param_hash

def get_git_hash():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__)
        ).decode().strip()
    except:
        return 'unknown'

config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'git_commit': get_git_hash()  # e.g., "a7b3c9d"
}

exp_hash = compute_param_hash(**config)
```

**Pros**:
- ✅ Automatic - no manual version bumping
- ✅ Precise - every code change gets unique hash
- ✅ Traceable - can checkout exact code version

**Cons**:
- ⚠️ Requires git repository
- ⚠️ Changes with EVERY commit (even docs/comments)
- ⚠️ May create too many versions

**Best for**: Research codebases with frequent iterations

---

### Solution 3: Semantic Versioning with Change Log

More structured approach with explicit change tracking:

```python
from embedding_tools import compute_param_hash

# Define implementation version separately
IMPLEMENTATION_VERSION = "2.1.0"
IMPLEMENTATION_CHANGELOG = {
    "2.1.0": "Fixed L2 normalization to use axis=1",
    "2.0.0": "Rewrote encoding pipeline with batching",
    "1.0.0": "Initial baseline implementation"
}

config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'implementation': {
        'version': IMPLEMENTATION_VERSION,
        'description': IMPLEMENTATION_CHANGELOG[IMPLEMENTATION_VERSION]
    }
}

exp_hash = compute_param_hash(**config)
```

**Pros**:
- ✅ Clear semantic meaning (major.minor.patch)
- ✅ Built-in documentation via changelog
- ✅ Easy to understand version differences

**Cons**:
- ⚠️ Requires manual maintenance
- ⚠️ Need discipline to update version

**Best for**: Production pipelines with release cycles

---

### Solution 4: Content-Based Hashing (Advanced)

Hash the actual code file to detect changes:

```python
import hashlib
from pathlib import Path
from embedding_tools import compute_param_hash

def hash_file(filepath):
    """Compute hash of Python file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'code_hash': hash_file(__file__)  # Hash this script
}

exp_hash = compute_param_hash(**config)
```

**Pros**:
- ✅ Fully automatic
- ✅ Detects ANY code change
- ✅ No manual version management

**Cons**:
- ⚠️ Changes on comment/whitespace edits
- ⚠️ Harder to understand what changed
- ⚠️ Hash looks cryptic in logs

**Best for**: Automated pipelines where reproducibility is critical

---

## Recommended Approach by Use Case

### Research / Experimentation
Use **Solution 1** (manual version) or **Solution 2** (git hash):

```python
config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'experiment_version': '1.0',  # Bump when changing algorithm
    'notes': 'Testing baseline with L2 normalization'
}
```

### Production Pipelines
Use **Solution 3** (semantic versioning):

```python
PIPELINE_VERSION = "2.1.0"  # At top of file

config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'pipeline_version': PIPELINE_VERSION
}
```

### Automated / CI/CD
Use **Solution 2** (git hash) or **Solution 4** (code hash):

```python
config = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'git_commit': get_git_hash(),
    'build_date': datetime.now().isoformat()
}
```

---

## Example: Semantic Search System

### Scenario
You're building a semantic search system that indexes documents and answers queries. You want to track both config changes AND code changes.

### Implementation

**semantic_search_system.py**:
```python
#!/usr/bin/env python
"""Semantic Search System with Document Embedding"""

from embedding_tools import compute_param_hash, EmbeddingStore, get_backend
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime

# Implementation version - bump when code changes
SYSTEM_VERSION = "1.0.0"
SYSTEM_CHANGELOG = {
    "1.0.0": "Initial semantic search with cosine similarity"
}

def build_search_index(documents, config_path='search_config.json'):
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Add implementation tracking
    config['system'] = {
        'version': SYSTEM_VERSION,
        'description': SYSTEM_CHANGELOG[SYSTEM_VERSION],
        'script': 'semantic_search_system.py'
    }

    # Compute experiment hash (includes system version)
    exp_hash = compute_param_hash(**config)
    print(f"Index hash: {exp_hash}")

    # Check if index exists
    index_file = f"indexes/search_index_{exp_hash}.pkl"
    if os.path.exists(index_file):
        print(f"Loading cached index from {index_file}")
        return load_index(index_file)

    # Build index
    print(f"Building search index v{SYSTEM_VERSION}...")

    # Load model
    model = SentenceTransformer(config['model']['name'])

    # Create embedding store
    backend = get_backend(config['backend']['type'])
    store = EmbeddingStore(
        backend=config['backend']['type'],
        max_memory_gb=config['backend']['max_memory_gb']
    )

    # Encode documents
    embeddings = encode_documents(documents, model, config)
    store.add_embeddings(
        embeddings,
        dimension=config['model']['dimension'],
        text_ids=[doc['id'] for doc in documents]
    )

    # Save with metadata
    index_data = {
        'store': store,
        'metadata': {
            'index_hash': exp_hash,
            'system_version': SYSTEM_VERSION,
            'config': config,
            'num_documents': len(documents),
            'timestamp': datetime.now().isoformat()
        }
    }
    save_index(index_data, index_file)

    return index_data

def encode_documents(documents, model, config):
    """Encode documents with batching and optional normalization."""
    texts = [doc['text'] for doc in documents]

    embeddings = model.encode(
        texts,
        batch_size=config['encoding']['batch_size'],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Apply normalization if configured
    if config['encoding'].get('normalize', False):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings
```

**search_config.json**:
```json
{
  "model": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "max_seq_length": 256
  },
  "backend": {
    "type": "mlx",
    "max_memory_gb": 10.0
  },
  "encoding": {
    "batch_size": 32,
    "normalize": true
  }
}
```

### Version Bump Example

**Bug Fix** (code changes):
```python
# Before (v1.0.0) - BUG: not normalizing embeddings when configured
def encode_documents(documents, model, config):
    texts = [doc['text'] for doc in documents]
    return model.encode(texts, batch_size=config['encoding']['batch_size'])
    # Bug: ignoring normalize setting!

# After (v1.0.1) - FIX: respect normalize configuration
def encode_documents(documents, model, config):
    texts = [doc['text'] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=config['encoding']['batch_size'],
        convert_to_numpy=True
    )

    # Fixed: now respects normalize setting
    if config['encoding'].get('normalize', False):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings

# Update version
SYSTEM_VERSION = "1.0.1"
SYSTEM_CHANGELOG = {
    "1.0.1": "Fixed: Respect normalize configuration setting",
    "1.0.0": "Initial semantic search with cosine similarity"
}
```

**Result**:
- Old indexes: hash includes `version: "1.0.0"` → cached with bug
- New indexes: hash includes `version: "1.0.1"` → rebuilds with fix
- Cache properly invalidated ✅

**Config Change** (no code change):
```json
{
  "encoding": {
    "batch_size": 64,  // Changed from 32
    "normalize": true
  }
}
```

**Result**:
- Hash changes automatically (batch_size in config)
- Version stays `1.0.1` (code unchanged)
- Separate index created ✅

---

## Pitfalls to Avoid

### ❌ Pitfall 1: Not Tracking Code Version
```python
# BAD: Only config parameters
config = {'model': 'all-MiniLM', 'dim': 384}
hash = compute_param_hash(**config)
# Problem: Code changes won't invalidate cache!
```

### ❌ Pitfall 2: Forgetting to Bump Version
```python
# BAD: Changed code but forgot to bump version
def encode(docs):
    return model.encode(docs) / np.linalg.norm(...)  # Changed algorithm

BASELINE_VERSION = "1.0.0"  # FORGOT TO BUMP!
# Problem: Same hash as before, will use old cached results
```

### ❌ Pitfall 3: Bumping Version for Non-Code Changes
```python
# BAD: Bumped version for comment change
BASELINE_VERSION = "1.0.1"  # Just added comments
# Problem: Unnecessary cache invalidation, wasted computation
```

### ✅ Good Practice: Use Constants at Top of File
```python
# GOOD: Clear versioning at top of file
BASELINE_VERSION = "1.2.0"
BASELINE_CHANGELOG = {
    "1.2.0": "Added early stopping in training",
    "1.1.0": "Switched to cosine similarity metric",
    "1.0.0": "Initial baseline"
}

# Easy to see current version and history
```

---

## Integration with embedding_tools

The `embedding_tools` library provides the hashing function, but **you** are responsible for:

1. ✅ **Deciding what to include in config**
   - Model parameters
   - Hyperparameters
   - **Implementation version** ← Your responsibility

2. ✅ **Bumping version when code changes**
   - Algorithm modifications
   - Bug fixes
   - Performance optimizations

3. ✅ **Documenting what changed**
   - CHANGELOG
   - Version notes in config
   - Git commit messages

**embedding_tools provides**:
- `compute_param_hash()` - Deterministic hashing
- `compute_config_hash()` - Dict-based hashing
- Order-independent hashing

**You provide**:
- Version numbers
- Change tracking
- Semantic meaning

---

## Future Enhancement (Potential Feature)

### Auto-Versioning Helper (Not Yet Implemented)

```python
# Future API idea - not currently available
from embedding_tools import VersionedExperiment

@VersionedExperiment(
    hash_code=True,        # Include code hash
    hash_dependencies=True # Include package versions
)
def run_experiment(config):
    # Your experiment code
    pass

# Would automatically:
# - Hash the function code
# - Include numpy, torch versions
# - Track git commit
# - Generate unique experiment ID
```

**Status**: Not implemented. Use manual versioning (Solutions 1-4 above) for now.

---

## Summary

### Key Takeaways

1. **Config hashing alone is NOT sufficient** for reproducibility
2. **Add a version parameter** to your config when code changes
3. **Use semantic versioning** (major.minor.patch) for clarity
4. **Document changes** in changelog or version notes
5. **Choose versioning strategy** based on your use case:
   - Research: Manual version or git hash
   - Production: Semantic versioning
   - Automated: Git hash or code hash

### Checklist for Experiment Versioning

- [ ] Add `version` or `pipeline_version` to config
- [ ] Define version at top of experiment script
- [ ] Bump version when changing:
  - [ ] Algorithm/logic
  - [ ] Bug fixes
  - [ ] Performance optimizations
- [ ] Don't bump version for:
  - [ ] Comments/documentation
  - [ ] Config-only changes (hash handles it)
  - [ ] Whitespace/formatting
- [ ] Document changes in CHANGELOG or version notes
- [ ] Include version in experiment hash via `compute_param_hash()`

---

## References

- **compute_param_hash()**: `embedding_tools/config/versioning.py`
- **Semantic Versioning**: https://semver.org/
- **Git commit hashing**: `git rev-parse --short HEAD`
- **Content-based hashing**: Python `hashlib.sha256()`

---

**Last Updated**: 2025-10-26
**Version**: 1.0
**Author**: Nitin Borwankar
