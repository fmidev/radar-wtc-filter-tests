# Tests for radar-wtc-filter-tests

This directory contains unit tests for the radar wind turbine clutter filter tests project.

## Running Tests

### With uv (recommended)

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_azimuth_masking.py

# Run with coverage report
uv run pytest --cov=src --cov-report=term-missing

# Run specific test class or method
uv run pytest tests/test_azimuth_masking.py::TestAzimuthWraparound
uv run pytest tests/test_azimuth_masking.py::TestAzimuthWraparound::test_wraparound_at_0_buffer_2
```

### Installing test dependencies

Test dependencies are defined in `pyproject.toml` under `[project.optional-dependencies]`:

```bash
uv pip install pytest pytest-cov
```

## Test Coverage

### test_azimuth_masking.py

Tests the azimuth buffer calculation and wraparound handling logic from `make_wtc_masks.py`.

**Test Classes:**

- `TestAzimuthBufferCalculation`: Tests normal cases without wraparound
  - Validates buffer sizes of 0, 1, 2, and 10
  - Ensures correct number of azimuths are masked

- `TestAzimuthWraparound`: Tests wraparound at 0°/360° boundary
  - Tests turbines at azimuths 0°, 1°, 358°, 359° with various buffers
  - Validates proper handling when buffer extends across the boundary
  
- `TestBufferSymmetry`: Parametrized tests for consistency
  - Tests 7 different azimuths (0, 1, 90, 180, 270, 358, 359)
  - Tests 5 different buffer sizes (0, 1, 2, 5, 10)
  - Validates that buffer size formula (2*buffer + 1) holds
  - Ensures turbine's own azimuth is always included

- `TestEdgeCases`: Tests unusual configurations
  - Different azimuth resolutions (720 bins, 10 bins)
  - Full circle coverage scenarios

**Total:** 56 test cases

## Continuous Integration

These tests ensure that:
1. The azimuth wraparound fix correctly handles the 0°/360° boundary
2. The off-by-one error fix produces correct buffer sizes
3. Future changes don't break existing functionality
4. The code works with different radar configurations

## Adding New Tests

When adding new functionality to `make_wtc_masks.py` or other source files:

1. Create a new test file in this directory: `test_<feature_name>.py`
2. Import pytest: `import pytest`
3. Organize tests into classes for related functionality
4. Use descriptive test names that explain what is being tested
5. Include docstrings explaining the test's purpose
6. Run tests to verify they pass before committing

Example:

```python
class TestNewFeature:
    """Test the new feature description."""
    
    def test_feature_basic_case(self):
        """Test that basic case works as expected."""
        result = my_function(input_data)
        assert result == expected_output
```
