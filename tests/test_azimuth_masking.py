"""Unit tests for azimuth buffer calculation and wraparound handling."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from make_wtc_masks import apply_turbine_mask


def get_masked_indices(mask):
    """
    Get set of azimuth indices that are masked for testing.
    
    Args:
        mask: 2D boolean mask array
        
    Returns:
        Set of azimuth indices where any range bin is True
    """
    # Get azimuths where at least one range bin is masked
    return set(np.where(mask.any(axis=1))[0])


def apply_azimuth_mask_for_test(a, buffer_azim, mask_shape_0):
    """
    Helper function to test azimuth masking logic.
    
    Creates a test mask and applies the turbine mask function,
    then returns which azimuth indices were masked.
    """
    # Create a test mask with arbitrary range dimension
    mask = np.zeros((mask_shape_0, 100), dtype=bool)
    
    # Apply the mask using the actual function
    # Use middle range so we don't hit range boundaries
    apply_turbine_mask(mask, 50, a, 10, 10, buffer_azim)
    
    return get_masked_indices(mask)


class TestAzimuthBufferCalculation:
    """Test azimuth buffer calculation without wraparound."""
    
    def test_normal_case_buffer_2(self):
        """Test turbine at 180° with buffer of 2."""
        result = apply_azimuth_mask_for_test(180, 2, 360)
        expected = {178, 179, 180, 181, 182}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_normal_case_buffer_1(self):
        """Test turbine at 100° with buffer of 1."""
        result = apply_azimuth_mask_for_test(100, 1, 360)
        expected = {99, 100, 101}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_normal_case_buffer_0(self):
        """Test turbine at 180° with buffer of 0 (no buffer)."""
        result = apply_azimuth_mask_for_test(180, 0, 360)
        expected = {180}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_normal_case_large_buffer(self):
        """Test turbine at 180° with large buffer of 10."""
        result = apply_azimuth_mask_for_test(180, 10, 360)
        expected = set(range(170, 191))
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 21, f"Expected 21 azimuths, got {len(result)}"


class TestAzimuthWraparound:
    """Test azimuth buffer calculation with wraparound at 0/360 boundary."""
    
    def test_wraparound_at_359_buffer_2(self):
        """Test turbine at 359° with buffer of 2."""
        result = apply_azimuth_mask_for_test(359, 2, 360)
        expected = {357, 358, 359, 0, 1}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 5, f"Expected 5 azimuths, got {len(result)}"
    
    def test_wraparound_at_0_buffer_2(self):
        """Test turbine at 0° with buffer of 2."""
        result = apply_azimuth_mask_for_test(0, 2, 360)
        expected = {358, 359, 0, 1, 2}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 5, f"Expected 5 azimuths, got {len(result)}"
    
    def test_wraparound_at_1_buffer_2(self):
        """Test turbine at 1° with buffer of 2."""
        result = apply_azimuth_mask_for_test(1, 2, 360)
        expected = {359, 0, 1, 2, 3}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 5, f"Expected 5 azimuths, got {len(result)}"
    
    def test_wraparound_at_358_buffer_2(self):
        """Test turbine at 358° with buffer of 2."""
        result = apply_azimuth_mask_for_test(358, 2, 360)
        expected = {356, 357, 358, 359, 0}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 5, f"Expected 5 azimuths, got {len(result)}"
    
    def test_wraparound_at_359_buffer_1(self):
        """Test turbine at 359° with buffer of 1."""
        result = apply_azimuth_mask_for_test(359, 1, 360)
        expected = {358, 359, 0}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 3, f"Expected 3 azimuths, got {len(result)}"
    
    def test_wraparound_at_0_buffer_1(self):
        """Test turbine at 0° with buffer of 1."""
        result = apply_azimuth_mask_for_test(0, 1, 360)
        expected = {359, 0, 1}
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 3, f"Expected 3 azimuths, got {len(result)}"
    
    def test_wraparound_at_0_buffer_0(self):
        """Test turbine at 0° with buffer of 0."""
        result = apply_azimuth_mask_for_test(0, 0, 360)
        expected = {0}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_wraparound_at_359_buffer_0(self):
        """Test turbine at 359° with buffer of 0."""
        result = apply_azimuth_mask_for_test(359, 0, 360)
        expected = {359}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_wraparound_large_buffer(self):
        """Test turbine at 5° with large buffer of 10 (wraps around)."""
        result = apply_azimuth_mask_for_test(5, 10, 360)
        expected = set(range(355, 360)) | set(range(0, 16))
        assert result == expected, f"Expected {expected}, got {result}"
        assert len(result) == 21, f"Expected 21 azimuths, got {len(result)}"


class TestBufferSymmetry:
    """Test that buffer is symmetric around turbine azimuth."""
    
    @pytest.mark.parametrize("azimuth", [0, 1, 90, 180, 270, 358, 359])
    @pytest.mark.parametrize("buffer", [0, 1, 2, 5, 10])
    def test_buffer_symmetry(self, azimuth, buffer):
        """Test that buffer size is correct: 2*buffer + 1."""
        result = apply_azimuth_mask_for_test(azimuth, buffer, 360)
        expected_size = 2 * buffer + 1
        assert len(result) == expected_size, \
            f"At azimuth {azimuth} with buffer {buffer}: " \
            f"expected {expected_size} azimuths, got {len(result)}"
    
    @pytest.mark.parametrize("azimuth", [0, 90, 180, 270, 359])
    def test_turbine_always_included(self, azimuth):
        """Test that the turbine's own azimuth is always in the mask."""
        for buffer in [0, 1, 2, 5]:
            result = apply_azimuth_mask_for_test(azimuth, buffer, 360)
            assert azimuth in result, \
                f"Turbine azimuth {azimuth} not in mask with buffer {buffer}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_full_circle_coverage(self):
        """Test that large buffer can cover entire circle."""
        result = apply_azimuth_mask_for_test(180, 180, 360)
        # Should cover all azimuths from 0 to 359
        assert len(result) == 360 or len(result) == 361, \
            f"Expected full coverage, got {len(result)} azimuths"
    
    def test_different_mask_shape(self):
        """Test with different azimuth resolution (e.g., 720 bins)."""
        result = apply_azimuth_mask_for_test(0, 2, 720)
        expected = {718, 719, 0, 1, 2}
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_small_mask_shape(self):
        """Test with small number of azimuth bins."""
        result = apply_azimuth_mask_for_test(0, 1, 10)
        expected = {9, 0, 1}
        assert result == expected, f"Expected {expected}, got {result}"
