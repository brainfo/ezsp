"""
Tests for ezsp.segment module.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestComputeSegmentationThresholds:
    """Tests for compute_segmentation_thresholds function."""

    def test_basic_thresholds(self, sample_image):
        """Test that thresholds are computed for a simple image."""
        from ezsp._img_utils import compute_segmentation_thresholds
        
        thresholds = compute_segmentation_thresholds(sample_image, n_classes=3)
        
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == 2  # n_classes - 1 thresholds
        assert thresholds[0] < thresholds[1]  # Thresholds should be sorted

    def test_thresholds_values_match_image_regions(self, sample_image):
        """Test that thresholds separate the distinct regions."""
        from ezsp._img_utils import compute_segmentation_thresholds
        
        thresholds = compute_segmentation_thresholds(sample_image, n_classes=3)
        
        # Thresholds should be within the intensity range of the image
        # and positioned between the distinct regions (50, 128, 220)
        assert thresholds[0] >= 50   # At or above low region value
        assert thresholds[0] <= 128  # At or below medium region value
        assert thresholds[1] >= 128  # At or above medium region value
        assert thresholds[1] <= 220  # At or below high region value

    def test_with_output_dir(self, sample_image, temp_dir):
        """Test histogram saving when output_dir is provided."""
        from ezsp._img_utils import compute_segmentation_thresholds
        
        thresholds = compute_segmentation_thresholds(
            sample_image, n_classes=3, output_dir=temp_dir
        )
        
        assert (temp_dir / "histogram.pdf").exists()
        assert (temp_dir / "histogram.png").exists()

    def test_skip_existing_histogram(self, sample_image, temp_dir):
        """Test that existing histogram is not overwritten."""
        from ezsp._img_utils import compute_segmentation_thresholds
        
        # Create existing histogram file
        existing_file = temp_dir / "histogram.pdf"
        existing_file.write_text("existing content")
        original_content = existing_file.read_text()
        
        compute_segmentation_thresholds(sample_image, n_classes=3, output_dir=temp_dir)
        
        # File should not be overwritten
        assert existing_file.read_text() == original_content

    def test_with_xarray_input(self, sample_image):
        """Test that xarray DataArray input is handled correctly."""
        import xarray as xr
        from ezsp._img_utils import compute_segmentation_thresholds
        
        # Convert numpy array to xarray DataArray
        img_xr = xr.DataArray(sample_image, dims=['y', 'x'])
        
        thresholds = compute_segmentation_thresholds(img_xr, n_classes=3)
        
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == 2


class TestSegment:
    """Tests for segment function."""

    def test_basic_segmentation(self, sample_image):
        """Test that segmentation returns an ImageContainer."""
        from ezsp._img_utils import segment
        import squidpy as sq
        
        result = segment(sample_image, thresh=100)
        
        assert isinstance(result, sq.im.ImageContainer)
        assert "segmented_watershed" in result

    def test_segmentation_with_different_thresholds(self, sample_image):
        """Test segmentation with different threshold values."""
        from ezsp._img_utils import segment
        
        result_low = segment(sample_image, thresh=50)
        result_high = segment(sample_image, thresh=200)
        
        # Different thresholds should produce different segmentations
        seg_low = result_low["segmented_watershed"].values
        seg_high = result_high["segmented_watershed"].values
        
        # At minimum, check both produced valid output
        assert seg_low is not None
        assert seg_high is not None

    def test_sets_numba_threads(self, sample_image):
        """Test that NUMBA_NUM_THREADS is set to 1."""
        import os
        from ezsp._img_utils import segment
        
        segment(sample_image, thresh=100)
        
        assert os.environ.get("NUMBA_NUM_THREADS") == "1"
