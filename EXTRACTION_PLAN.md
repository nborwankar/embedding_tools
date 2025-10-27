# Extraction Plan: embedding_tools → Standalone Library

**Generated**: October 25, 2025
**Status**: Planning Phase

---

## Current State Assessment ✅

**Good News:**
- ✅ **Zero back-dependencies**: embedding_tools has NO imports from writeapaper
- ✅ **Self-contained**: All code is internal to the package
- ✅ **Proper packaging**: Has complete pyproject.toml with setuptools
- ✅ **Minimal consumers**: Only `kb_tree_matryoshka/experiments/msmarco/baseline_1024d.py` uses it
- ✅ **No path hacks**: No sys.path manipulation or PYTHONPATH dependencies

**Current Location**: `/Users/nitin/Projects/github/writeapaper/other/embedding_tools/`

**Consumer Code**: `github/writeapaper/kb_tree_matryoshka/experiments/msmarco/baseline_1024d.py`

---

## Phase 1: Pre-Flight Checks 🔍

### 1.1 Verify Current Installation State
```bash
# Check if already installed
pip list | grep embedding

# Check current imports work
cd github/writeapaper/kb_tree_matryoshka/experiments/msmarco/
python -c "from embedding_tools import get_backend; print('✓ Import works')"
```

### 1.2 Document Current Consumers
```bash
# Find all files importing embedding_tools
grep -r "from embedding_tools\|import embedding_tools" github/writeapaper --include="*.py" > /tmp/embedding_tools_consumers.txt

# Expected: Only baseline_1024d.py (if it exists)
```

### 1.3 Check for Test Data Dependencies
```bash
# Verify tests don't depend on writeapaper data
cd github/writeapaper/other/embedding_tools
pytest tests/ --collect-only  # Should list all tests without errors
```

---

## Phase 2: Extract to New Location 📦

### 2.1 Choose New Location

**Option A (Recommended)**: Neutral GitHub-like location
```
/Users/nitin/Projects/github/embedding_tools/
```
**Pros**: Consistent with other standalone projects, ready for GitHub push
**Cons**: None

**Option B**: Top-level Projects
```
/Users/nitin/Projects/embedding_tools/
```
**Pros**: Simpler path
**Cons**: Different pattern than other standalone code

**Recommendation**: Use **Option A** (`github/embedding_tools/`)

### 2.2 Copy (Don't Move Yet)
```bash
# Copy entire directory to new location
cp -r github/writeapaper/other/embedding_tools github/embedding_tools

# Verify structure
ls -la github/embedding_tools/
```

---

## Phase 3: Update Metadata 📝

### 3.1 Update pyproject.toml
**Changes needed in** `github/embedding_tools/pyproject.toml`:

```toml
# OLD
authors = [
    {name = "WriteAPaper Project"},
]

[project.urls]
Homepage = "https://github.com/writeapaper/embedding_tools"
Repository = "https://github.com/writeapaper/embedding_tools"
Issues = "https://github.com/writeapaper/embedding_tools/issues"

# NEW
authors = [
    {name = "Nitin Borwankar", email = "nborwankar@gmail.com"},
]

[project.urls]
Homepage = "https://github.com/nborwankar/embedding_tools"  # or your GitHub
Repository = "https://github.com/nborwankar/embedding_tools"
Issues = "https://github.com/nborwankar/embedding_tools/issues"
```

### 3.2 Update README.md
**Add to top of** `github/embedding_tools/README.md`:

```markdown
# embedding_tools

> **Note**: This library was extracted from the WriteAPaper research project
> and is now maintained as a standalone package.

[![PyPI version](https://badge.fury.io/py/embedding_tools.svg)](https://badge.fury.io/py/embedding_tools)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### 3.3 Create .gitignore (if not exists)
```bash
cat > github/embedding_tools/.gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.env
.venv
env/
venv/
EOF
```

---

## Phase 4: Install as Editable Package 🔧

### 4.1 Uninstall Old Version (if exists)
```bash
pip uninstall embedding_tools -y
```

### 4.2 Install from New Location
```bash
# Install in editable/development mode
cd github/embedding_tools
pip install -e .

# Verify installation
pip show embedding_tools
# Should show: Location: /Users/nitin/Projects/github/embedding_tools
```

### 4.3 Install Optional Dependencies
```bash
# Install MLX support (for Apple Silicon)
pip install -e ".[mlx]"

# Or install everything
pip install -e ".[all]"
```

---

## Phase 5: Update Consumer Code 🔄

### 5.1 Update kb_tree_matryoshka imports

**File**: `github/writeapaper/kb_tree_matryoshka/experiments/msmarco/baseline_1024d.py`

**Current** (probably):
```python
from embedding_tools import (
    get_backend,
    EmbeddingStore,
    compute_config_hash
)
```

**After extraction**: **No change needed!**
- Because we installed via `pip install -e .`, Python will find it automatically
- Import statement stays exactly the same

### 5.2 Remove sys.path Hacks (if any exist)

Check `baseline_1024d.py` for any lines like:
```python
# OLD - Remove if found
import sys
sys.path.insert(0, '../../other/embedding_tools')
```

---

## Phase 6: Testing 🧪

### 6.1 Test Standalone Library
```bash
cd github/embedding_tools

# Run all tests
pytest tests/ -v

# Run validation script
python validate.py

# Expected: 52 tests pass, 5 validation checks ✓
```

### 6.2 Test Consumer Integration
```bash
cd github/writeapaper/kb_tree_matryoshka/experiments/msmarco/

# Test import works
python -c "from embedding_tools import get_backend; print(f'Backend: {get_backend()}')"

# If baseline_1024d.py exists, test it
python baseline_1024d.py --test  # or whatever test mode exists
```

### 6.3 Test in Fresh Environment
```bash
# Create test venv
python -m venv /tmp/test_embedding_tools
source /tmp/test_embedding_tools/bin/activate

# Install from new location
pip install -e /Users/nitin/Projects/github/embedding_tools

# Test
python -c "from embedding_tools import get_backend; print('✓ Works in fresh env')"

deactivate
```

---

## Phase 7: Git Management 🗂️

### 7.1 Initialize New Repo
```bash
cd github/embedding_tools

# Initialize git
git init

# Create initial commit
git add .
git commit -m "Initial commit: Extract embedding_tools from WriteAPaper project

Extracted from github/writeapaper/other/embedding_tools/ on 2025-10-25.

Features:
- Cross-platform backend abstraction (NumPy, MLX, PyTorch)
- Memory-managed EmbeddingStore for large embedding sets
- Configuration versioning with SHA-256 hashing
- 52 passing tests

Originally developed for MS MARCO experiments in KB Tree Matryoshka paper."
```

### 7.2 Handle Old Location

**Option A** (Recommended): Keep as deprecated reference
```bash
cd github/writeapaper/other/embedding_tools

# Create README pointing to new location
cat > MOVED.md << 'EOF'
# embedding_tools has moved!

This library has been extracted to a standalone location:

**New Location**: `/Users/nitin/Projects/github/embedding_tools/`

**Installation**:
```bash
pip install -e /Users/nitin/Projects/github/embedding_tools
```

**Old imports still work** - no code changes needed in your projects.

Last synchronized: 2025-10-25
EOF

git add MOVED.md
git commit -m "Add notice: embedding_tools extracted to standalone repo"
```

**Option B**: Remove old directory (do later, after thorough testing)
```bash
# DON'T DO THIS YET - only after Phase 8 is complete
cd github/writeapaper/other
git rm -rf embedding_tools
git commit -m "Remove embedding_tools (now standalone at github/embedding_tools/)"
```

---

## Phase 8: Validation Checklist ✅

### 8.1 Standalone Library Checks
- [ ] Tests pass: `pytest tests/ -v` (52/52)
- [ ] Validation passes: `python validate.py` (5/5 checks)
- [ ] Installation works: `pip install -e .`
- [ ] Documentation builds (if applicable)
- [ ] No references to writeapaper in code
- [ ] pyproject.toml has correct URLs

### 8.2 Consumer Code Checks
- [ ] `baseline_1024d.py` imports work
- [ ] MS MARCO experiments still run
- [ ] No broken imports in writeapaper projects
- [ ] No PYTHONPATH or sys.path hacks needed

### 8.3 Cross-Platform Checks
- [ ] NumPy backend works (always)
- [ ] MLX backend works (M2 Mac)
- [ ] PyTorch MPS backend works (M2 Mac)
- [ ] Tests pass on clean venv

---

## Phase 9: Documentation Updates 📚

### 9.1 Update embedding_tools README
Add installation section:
```markdown
## Installation

### From Local Development Copy
```bash
pip install -e /Users/nitin/Projects/github/embedding_tools
```

### From PyPI (future)
```bash
pip install embedding_tools
```
```

### 9.2 Update WriteAPaper Documentation
Update `github/writeapaper/other/CLAUDE.md`:
```markdown
## embedding_tools Library

**Status**: Extracted to standalone repository

**Location**: `/Users/nitin/Projects/github/embedding_tools/`

**Installation**:
```bash
pip install -e /Users/nitin/Projects/github/embedding_tools
```

**Usage**: Import normally - pip installation makes it globally available.
```

### 9.3 Create CHANGELOG.md
```markdown
# Changelog

## [0.1.0] - 2025-10-25

### Added
- Extracted from WriteAPaper project as standalone library
- Cross-platform backend abstraction (NumPy, MLX, PyTorch)
- EmbeddingStore with memory management
- Configuration versioning
- 52 comprehensive tests
```

---

## Phase 10: Optional - Publish to PyPI 🚀 (Future)

### 10.1 Prepare for PyPI
```bash
cd github/embedding_tools

# Build distribution
python -m build

# Test installation from dist
pip install dist/embedding_tools-0.1.0-py3-none-any.whl
```

### 10.2 Publish to Test PyPI (dry run)
```bash
# Install twine
pip install twine

# Upload to test.pypi.org
twine upload --repository testpypi dist/*

# Test install from test PyPI
pip install --index-url https://test.pypi.org/simple/ embedding_tools
```

### 10.3 Publish to Real PyPI
```bash
# Only when ready!
twine upload dist/*
```

---

## Summary: Extraction Complexity

**Difficulty**: ⭐⭐☆☆☆ (Low-Medium)

**Time Estimate**: 1-2 hours

**Risk Level**: Low
- No code changes required in consumers
- No back-dependencies to break
- Proper packaging already exists

**Key Success Factors**:
1. Install via `pip install -e .` so imports "just work"
2. Keep old location temporarily with MOVED.md notice
3. Test thoroughly before removing old directory
4. Document the extraction in both repos

**Recommended Timeline**:
- **Now**: Phases 1-6 (extraction + testing)
- **After 1 week**: Phase 7.2 Option B (remove old location if all works)
- **Future**: Phase 10 (PyPI publication if desired)

---

## Notes

- This plan assumes minimal disruption to existing workflows
- All consumer code continues to work without modification
- Old location can remain indefinitely as a deprecated reference
- PyPI publication is optional and can be done anytime

**Next Step**: Review this plan and proceed with Phase 1 when ready.
