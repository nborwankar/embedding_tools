# PyTorch Backend Fix

**Date**: 2025-10-26
**Issue**: Type hint bug preventing package import
**Status**: ✅ Fixed

---

## Problem

### Symptom
```python
from embedding_tools import get_backend

# Error:
NameError: name 'torch' is not defined
```

### Root Cause

In `embedding_tools/arrays/torch_backend.py`, type hints used `torch.Tensor` directly:

```python
def create_array(self, data: Any, dtype: Optional[str] = None) -> torch.Tensor:
    ...
```

**Issue**: Type hints are evaluated at **class definition time**, not runtime.

Even though torch was imported in a try-except block:
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

The type hint `-> torch.Tensor` was evaluated BEFORE the try-except executed, causing:
1. If torch import fails → `NameError: name 'torch' is not defined`
2. If torch import succeeds but is broken → Same error when torch module loads

This **broke ALL imports** of embedding_tools, even NumPy and MLX backends!

---

## Solution

Added `from __future__ import annotations` to make all type hints strings (PEP 563):

```python
# Before
"""PyTorch backend implementation..."""

from typing import Any, List, Optional, Tuple
import numpy as np

# After
"""PyTorch backend implementation..."""

from __future__ import annotations  # NEW
from typing import Any, List, Optional, Tuple
import numpy as np
```

**Effect**: All type hints are now treated as strings and evaluated lazily, preventing early evaluation errors.

---

## Impact

### Before Fix
- ❌ Cannot import embedding_tools at all
- ❌ NumPy backend unusable (import fails)
- ❌ MLX backend unusable (import fails)
- ❌ Consumer (baseline_1024d.py) completely broken

### After Fix
- ✅ NumPy backend works
- ✅ MLX backend works
- ✅ PyTorch backend fails gracefully if torch broken
- ✅ Consumer imports successfully
- ✅ Package usable even with broken PyTorch installation

---

## Additional PyTorch Issue

**Separate Problem**: PyTorch installation is corrupted

```
ImportError: dlopen(...torch/_C.cpython-311-darwin.so, 0x0002):
Library not loaded: @rpath/libtorch_cpu.dylib
```

**Status**: Not fixed (but doesn't block embedding_tools anymore)

**Workaround**: Use NumPy or MLX backend:
```python
# Avoid PyTorch
backend = get_backend('mlx')   # Use MLX on Mac
backend = get_backend('numpy')  # Use NumPy anywhere
```

**Fix** (if needed later):
```bash
# Reinstall PyTorch
pip uninstall torch -y
pip install torch
```

---

## Testing Results

After fix, all core functionality works:

### ✅ NumPy Backend
```
✓ NumPy backend imported successfully
✓ create_array: shape=(2, 3)
✓ cosine_similarity: [1. 0.]
✅ NumPy backend: ALL TESTS PASSED
```

### ✅ MLX Backend
```
✓ MLX backend imported successfully
✓ Device: M2 Mac
✓ create_array: shape=(2, 3)
✓ cosine_similarity: [1. 0.]
✓ memory_usage: 24 bytes
✅ MLX backend: ALL TESTS PASSED
```

### ✅ Full Package
```
✓ Package imports successful
✓ Auto-detected backend: MLXBackend
✓ NumPy backend: NumpyBackend
✓ MLX backend: MLXBackend
✓ EmbeddingStore works with 100 embeddings
✓ compute_param_hash: a478ea550cac49a0
✅ FULL PACKAGE: ALL TESTS PASSED
```

### ✅ Consumer Integration
```
✓ All imports from consumer successful
✓ detect_best_backend available
✓ detect_best_device available
✓ get_device_info available
✓ EmbeddingStore available
✓ compute_param_hash available
✅ CONSUMER CAN IMPORT embedding_tools
```

---

## Lessons Learned

1. **Type hints in optional backends**: Use `from __future__ import annotations` when type hints reference optionally-imported modules

2. **Guard pattern is insufficient**: Try-except around imports doesn't protect type hints

3. **Better pattern for optional dependencies**:
   ```python
   from __future__ import annotations  # Make hints strings
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       import torch  # Only for type checkers

   try:
       import torch
       AVAILABLE = True
   except ImportError:
       AVAILABLE = False

   def method(self, data) -> torch.Tensor:  # Now safe
       ...
   ```

4. **Test without optional dependencies**: Should test with `pip install embedding_tools` (core only) to catch these issues

---

## Files Changed

- `embedding_tools/arrays/torch_backend.py`: Added `from __future__ import annotations`

**Diff**:
```diff
  """PyTorch backend implementation with CUDA/MPS/CPU support."""

+ from __future__ import annotations
  from typing import Any, List, Optional, Tuple
  import numpy as np
```

**Lines changed**: 1 line added (line 7)

---

## Recommendations

### For Current State
- ✅ Fix is sufficient for now
- ✅ NumPy and MLX backends fully functional
- ⚠️ PyTorch backend broken but doesn't impact other backends

### For Future
1. **Add same fix to JAX backend** when implementing (preventive)
2. **Test suite**: Add test that imports package without optional dependencies installed
3. **CI/CD**: Test with `pip install embedding_tools` (no extras)
4. **Fix PyTorch**: Reinstall when needed for Linux production

---

## PyTorch Installation Fix (2025-10-26)

**Solution**: Created dedicated conda environment

The corrupted PyTorch installation was resolved by creating a clean conda environment:

```bash
# Create conda environment
conda create -n embedding_tools python=3.11 -y

# Activate environment
conda activate embedding_tools

# Install embedding_tools with all dependencies (including PyTorch)
pip install -e ".[all]"
```

### Installation Results

**Packages Installed**:
- numpy 2.3.4
- mlx 0.29.3
- mlx-metal 0.29.3
- **torch 2.9.0** ✅
- All dependencies (sympy, networkx, filelock, etc.)

### Testing Results (All Passing)

**NumPy Backend**:
```
✓ Backend: NumpyBackend
✓ Created array: shape=(2, 3)
✓ Cosine similarity: [[0.9999999 0.9746318]]
```

**MLX Backend**:
```
✓ Backend: MLXBackend
✓ Created array: shape=(2, 3)
✓ Cosine similarity: [[0.9999999 0.9746318]]
```

**PyTorch Backend** ✅ NOW WORKING:
```
TorchBackend using device: mps
✓ Backend: TorchBackend
✓ Device: mps
✓ Created array: shape=(2, 3)
✓ Cosine similarity: [0.99999994 0.9746318]
```

**Full Validation Suite**:
```
[1/5] Testing package import... ✓
[2/5] Testing NumPy backend... ✓
[3/5] Testing MLX backend... ✓
[4/5] Testing EmbeddingStore... ✓
[5/5] Testing configuration versioning... ✓
```

**PyTorch-Specific Tests** (7 tests):
```
[1] Auto-detection: mps ✓
[2] Explicit MPS: mps ✓
[3] Basic operations: ✓
[4] Cosine similarity: ✓
[5] Dimension slicing: ✓
[6] EmbeddingStore integration: ✓
[7] Memory info: ✓
```

---

## Final Status

**Working Backends**:
- ✅ NumPy (CPU)
- ✅ MLX (Apple Silicon GPU)
- ✅ **PyTorch (MPS - Apple Silicon GPU)** 🎉

**Broken Backend**:
- None! All backends working.

**Package Status**:
- ✅ Clean conda environment created
- ✅ PyTorch 2.9.0 installed successfully
- ✅ All backends tested and working
- ✅ Full validation suite passing
- ✅ PyTorch using MPS device (Apple Silicon GPU acceleration)
- ✅ Ready for production use

**Development Environment**:
```bash
# Activate environment for development
conda activate embedding_tools

# Run tests
python validate.py
python test_torch_backend.py
pytest tests/ -v
```

**Next Step**: This issue is now fully resolved. PyTorch backend is production-ready.
