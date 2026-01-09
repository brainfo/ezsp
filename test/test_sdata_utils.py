"""
Tests for ezsp.sdata_utils module.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSafeUpdateSdata:
    """Tests for safe_update_sdata function."""

    def test_successful_write_and_verify(self, temp_dir):
        """Test successful write and verification workflow."""
        from ezsp._io import safe_update_sdata
        
        # Create mock SpatialData object
        mock_sdata = MagicMock()
        mock_sdata.images.keys.return_value = ['image1']
        mock_sdata.labels.keys.return_value = ['label1']
        mock_sdata.shapes.keys.return_value = ['shape1']
        mock_sdata.tables.keys.return_value = ['table1']
        
        # Create mock for read back verification
        mock_verify = MagicMock()
        mock_verify.images.keys.return_value = ['image1']
        mock_verify.labels.keys.return_value = ['label1']
        mock_verify.shapes.keys.return_value = ['shape1']
        mock_verify.tables.keys.return_value = ['table1']
        
        new_path = temp_dir / "new_sdata.zarr"
        
        with patch('ezsp._io.sd.read_zarr', return_value=mock_verify):
            result = safe_update_sdata(mock_sdata, new_path)
        
        # Verify write was called
        mock_sdata.write.assert_called_once_with(new_path)
        
        # Verify result is the verified sdata
        assert result == mock_verify

    def test_verification_failure_raises_assertion(self, temp_dir):
        """Test that mismatched keys raise AssertionError."""
        from ezsp._io import safe_update_sdata
        
        mock_sdata = MagicMock()
        mock_sdata.images.keys.return_value = ['image1', 'image2']
        mock_sdata.labels.keys.return_value = []
        mock_sdata.shapes.keys.return_value = []
        mock_sdata.tables.keys.return_value = []
        
        mock_verify = MagicMock()
        mock_verify.images.keys.return_value = ['image1']  # Missing image2!
        mock_verify.labels.keys.return_value = []
        mock_verify.shapes.keys.return_value = []
        mock_verify.tables.keys.return_value = []
        
        new_path = temp_dir / "new_sdata.zarr"
        
        with patch('ezsp._io.sd.read_zarr', return_value=mock_verify):
            with pytest.raises(AssertionError):
                safe_update_sdata(mock_sdata, new_path)

    def test_old_path_deleted_after_verification(self, temp_dir):
        """Test that old_path is deleted after successful verification."""
        from ezsp._io import safe_update_sdata
        
        mock_sdata = MagicMock()
        mock_sdata.images.keys.return_value = []
        mock_sdata.labels.keys.return_value = []
        mock_sdata.shapes.keys.return_value = []
        mock_sdata.tables.keys.return_value = []
        
        mock_verify = MagicMock()
        mock_verify.images.keys.return_value = []
        mock_verify.labels.keys.return_value = []
        mock_verify.shapes.keys.return_value = []
        mock_verify.tables.keys.return_value = []
        
        new_path = temp_dir / "new_sdata.zarr"
        old_path = temp_dir / "old_sdata.zarr"
        old_path.mkdir()  # Create old directory
        
        with patch('ezsp._io.sd.read_zarr', return_value=mock_verify):
            safe_update_sdata(mock_sdata, new_path, old_path=old_path)
        
        # Old path should be deleted
        assert not old_path.exists()

    def test_old_path_not_deleted_if_not_exists(self, temp_dir):
        """Test that non-existent old_path doesn't cause errors."""
        from ezsp._io import safe_update_sdata
        
        mock_sdata = MagicMock()
        mock_sdata.images.keys.return_value = []
        mock_sdata.labels.keys.return_value = []
        mock_sdata.shapes.keys.return_value = []
        mock_sdata.tables.keys.return_value = []
        
        mock_verify = MagicMock()
        mock_verify.images.keys.return_value = []
        mock_verify.labels.keys.return_value = []
        mock_verify.shapes.keys.return_value = []
        mock_verify.tables.keys.return_value = []
        
        new_path = temp_dir / "new_sdata.zarr"
        old_path = temp_dir / "nonexistent.zarr"
        
        with patch('ezsp._io.sd.read_zarr', return_value=mock_verify):
            # Should not raise even if old_path doesn't exist
            result = safe_update_sdata(mock_sdata, new_path, old_path=old_path)
        
        assert result == mock_verify

    def test_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        from ezsp._io import safe_update_sdata
        
        mock_sdata = MagicMock()
        mock_sdata.images.keys.return_value = []
        mock_sdata.labels.keys.return_value = []
        mock_sdata.shapes.keys.return_value = []
        mock_sdata.tables.keys.return_value = []
        
        mock_verify = MagicMock()
        mock_verify.images.keys.return_value = []
        mock_verify.labels.keys.return_value = []
        mock_verify.shapes.keys.return_value = []
        mock_verify.tables.keys.return_value = []
        
        new_path_str = str(temp_dir / "new_sdata.zarr")
        
        with patch('ezsp._io.sd.read_zarr', return_value=mock_verify):
            result = safe_update_sdata(mock_sdata, new_path_str)
        
        # Should work with string path
        assert result == mock_verify
