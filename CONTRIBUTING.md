# Contributing to embedding_tools

Thank you for your interest in contributing to embedding_tools! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/embedding_tools.git
cd embedding_tools
```

### 2. Set Up Development Environment

```bash
# Create a conda environment
conda create -n embedding_tools python=3.11 -y
conda activate embedding_tools

# Install in development mode with all extras
pip install -e ".[dev,all]"

# Verify installation
python validate.py
pytest tests/ -v
```

### 3. Create a Branch

```bash
git checkout -b my-feature-branch
```

## Making Changes

### Code Standards

- **Formatting**: Use Black with 100-character line length
  ```bash
  black embedding_tools/ tests/ examples/
  ```

- **Import Sorting**: Use isort with Black profile
  ```bash
  isort embedding_tools/ tests/ examples/
  ```

- **Linting**: Code must pass flake8
  ```bash
  flake8 embedding_tools/
  ```

- **Type Hints**: Add type hints to new functions/methods

- **Docstrings**: Use Google-style docstrings for public APIs
  ```python
  def cosine_similarity(query, embeddings):
      """Compute cosine similarity between query and embeddings.

      Args:
          query: Query vector of shape (d,) or (1, d)
          embeddings: Embedding matrix of shape (n, d)

      Returns:
          Similarity scores of shape (n,)

      Raises:
          ValueError: If dimensions don't match
      """
  ```

### Testing Requirements

All contributions must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_arrays.py -v

# Run with coverage
pytest tests/ --cov=embedding_tools
```

**Test Guidelines**:
- Add tests for new features in appropriate test file
- Ensure edge cases are covered
- Test error conditions (invalid inputs, etc.)
- Maintain or improve code coverage

### Backend Compatibility

**CRITICAL**: If you modify `ArrayBackend` operations, you MUST update ALL backends:

1. **Modify**: `embedding_tools/arrays/base.py` (abstract method)
2. **Implement**: `embedding_tools/arrays/numpy_backend.py`
3. **Implement**: `embedding_tools/arrays/mlx_backend.py`
4. **Implement**: `embedding_tools/arrays/torch_backend.py`
5. **Test**: Add tests in `tests/test_arrays.py` for all three backends

Example:
```python
# In base.py
class ArrayBackend(ABC):
    @abstractmethod
    def my_new_operation(self, array):
        """Description of operation."""
        pass

# In numpy_backend.py
def my_new_operation(self, array):
    return np.my_operation(array)

# In mlx_backend.py
def my_new_operation(self, array):
    return mx.my_operation(array)

# In torch_backend.py
def my_new_operation(self, array):
    return torch.my_operation(array)

# In test_arrays.py
def test_my_new_operation_numpy():
    backend = NumpyBackend()
    result = backend.my_new_operation(test_array)
    assert expected_condition

def test_my_new_operation_mlx():
    # Similar test for MLX
    pass

def test_my_new_operation_torch():
    # Similar test for PyTorch
    pass
```

## Submitting Changes

### 1. Pre-Submission Checklist

Before opening a PR, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black . && isort .`
- [ ] No lint errors: `flake8 embedding_tools/`
- [ ] New tests added for new functionality
- [ ] Documentation updated (docstrings, README.md if needed)
- [ ] CHANGELOG.md updated under "Unreleased" section
- [ ] All backends updated if modifying ArrayBackend

### 2. Commit Your Changes

```bash
git add .
git commit -m "Add feature: brief description"
```

Use clear, descriptive commit messages:
- ✅ "Add euclidean distance to ArrayBackend"
- ✅ "Fix memory leak in EmbeddingStore.slice_to_dimension"
- ❌ "Fix bug"
- ❌ "Update code"

### 3. Push and Create PR

```bash
git push origin my-feature-branch
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference any related issues: "Fixes #123"

### 4. PR Review Process

- Maintainer will review your PR
- Address any requested changes
- Once approved, PR will be merged (usually with squash)

## Types of Contributions

### Bug Fixes

1. Open an issue describing the bug (if not already open)
2. Create a branch: `git checkout -b fix/bug-description`
3. Fix the bug and add a test that would have caught it
4. Submit PR referencing the issue

### New Features

1. Open an issue to discuss the feature first
2. Wait for maintainer approval (ensures alignment with project goals)
3. Implement the feature with tests
4. Update documentation
5. Submit PR

### Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples
- Improve docstrings
- Update guides

No issue required for documentation PRs.

### Performance Improvements

1. Open an issue with benchmark results showing the improvement
2. Ensure changes don't break existing functionality
3. Add benchmarks if not already present
4. Submit PR with before/after performance metrics

## Development Tips

### Running Examples

```bash
# Test all examples work
python examples/basic_usage.py
python examples/device_detection_workflow.py
python examples/config_driven_backend.py
```

### Testing on Different Backends

```bash
# Test with only NumPy (no MLX/PyTorch)
pip uninstall mlx torch -y
pytest tests/ -v

# Reinstall all backends
pip install -e ".[all]"
```

### Debugging Tests

```bash
# Run specific test with verbose output
pytest tests/test_arrays.py::TestNumpyBackend::test_cosine_similarity -v -s

# Drop into debugger on failure
pytest tests/ --pdb
```

## Code of Conduct

### Expected Behavior

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: nborwankar@gmail.com

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Feature ideas**: Open a GitHub Issue for discussion first
- **Security issues**: Email nborwankar@gmail.com (do not open public issue)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make embedding_tools better for everyone. We appreciate your time and effort!

---

**Additional Resources**:
- [Maintenance Guide](docs/MAINTENANCE.md) - For maintainers
- [README.md](README.md) - Project overview
- [CHANGELOG.md](docs/CHANGELOG.md) - Version history
