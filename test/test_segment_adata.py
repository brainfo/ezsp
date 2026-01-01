"""
Tests for ezsp.segment_adata module.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_log_directory(self, temp_dir):
        """Test that log directory is created if it doesn't exist."""
        from ezsp.segment_adata import setup_logging
        
        log_dir = temp_dir / "new_logs"
        assert not log_dir.exists()
        
        setup_logging(str(log_dir))
        
        assert log_dir.exists()

    def test_returns_log_file_path(self, temp_dir):
        """Test that setup_logging returns the log file path."""
        from ezsp.segment_adata import setup_logging
        
        log_file = setup_logging(str(temp_dir))
        
        assert isinstance(log_file, Path)
        assert log_file.parent == temp_dir
        assert log_file.suffix == ".log"
        assert "tangram_" in log_file.name


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_yaml_config(self, temp_dir, sample_config):
        """Test loading a valid YAML config file."""
        import yaml
        from ezsp.segment_adata import load_config
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        
        result = load_config(str(config_file))
        
        assert result == sample_config

    def test_loads_batch_config(self, temp_dir, sample_batch_config):
        """Test loading a batch configuration with lists."""
        import yaml
        from ezsp.segment_adata import load_config
        
        config_file = temp_dir / "batch_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_batch_config, f)
        
        result = load_config(str(config_file))
        
        assert isinstance(result["sdata_path"], list)
        assert len(result["sdata_path"]) == 2


class TestPrepareForSquidpy:
    """Tests for prepare_for_squidpy function."""

    def test_creates_spatial_metadata(self, mock_adata, sample_rgb_image):
        """Test that spatial metadata is properly created."""
        from ezsp.segment_adata import prepare_for_squidpy
        
        # Create mock SpatialData
        mock_sdata = MagicMock()
        mock_sdata.tables = {"bin100_table": mock_adata}
        
        # Mock multiscale image access
        mock_image = MagicMock()
        mock_image.__getitem__ = lambda self, key: MagicMock(
            __getitem__=lambda self2, key2: MagicMock(
                values=np.moveaxis(sample_rgb_image, -1, 0)  # CHW format
            )
        )
        mock_sdata.images = {"stain": mock_image}
        
        library_id = "test_sample"
        
        with patch('ezsp.segment_adata.normalize_image', side_effect=lambda x, compute: x):
            result = prepare_for_squidpy(
                mock_sdata,
                image_key="stain",
                library_id=library_id,
                table_key="bin100_table",
            )
        
        assert "spatial" in result.uns
        assert library_id in result.uns["spatial"]
        assert "images" in result.uns["spatial"][library_id]
        assert "scalefactors" in result.uns["spatial"][library_id]


class TestSegmentAndExtractFeatures:
    """Tests for segment_and_extract_features function."""

    def test_adds_cell_count(self, mock_adata_with_spatial):
        """Test that cell_count is added to adata.obs."""
        from ezsp.segment_adata import segment_and_extract_features
        
        adata = mock_adata_with_spatial.copy()
        library_id = "test_sample"
        
        # This test would require actual squidpy functionality
        # For unit testing, we mock the squidpy calls
        with patch('ezsp.segment_adata.sq.im.ImageContainer') as mock_imgc:
            with patch('ezsp.segment_adata.sq.im.segment'):
                with patch('ezsp.segment_adata.sq.im.calculate_image_features') as mock_calc:
                    # Setup mock to add required obsm data
                    def add_features(adata, *args, **kwargs):
                        adata.obsm["image_features"] = pd.DataFrame({
                            "segmentation_label": np.random.randint(1, 10, adata.n_obs)
                        }, index=adata.obs_names)
                    mock_calc.side_effect = add_features
                    
                    result = segment_and_extract_features(
                        adata, library_id=library_id, thresh=100
                    )
        
        assert "cell_count" in result.obs.columns


class TestPrepareSingleCellReference:
    """Tests for prepare_single_cell_reference function."""

    def test_filters_cell_types(self, mock_sc_adata, temp_dir):
        """Test that cell types with too few cells are filtered."""
        from ezsp.segment_adata import prepare_single_cell_reference
        
        # Save mock data to file
        sc_file = temp_dir / "sc_ref.h5ad"
        mock_sc_adata.write_h5ad(sc_file)
        
        adata_sc, markers = prepare_single_cell_reference(
            str(sc_file),
            n_top_genes=10,
            min_cells_per_type=5,
        )
        
        # All remaining cell types should have >= 5 cells
        celltype_counts = adata_sc.obs["major_celltype"].value_counts()
        assert all(count >= 5 for count in celltype_counts)

    def test_returns_marker_genes(self, mock_sc_adata, temp_dir):
        """Test that marker genes are identified."""
        from ezsp.segment_adata import prepare_single_cell_reference
        
        sc_file = temp_dir / "sc_ref.h5ad"
        mock_sc_adata.write_h5ad(sc_file)
        
        adata_sc, markers = prepare_single_cell_reference(
            str(sc_file),
            n_top_genes=10,
            min_cells_per_type=5,
        )
        
        assert isinstance(markers, list)
        assert len(markers) > 0
        assert all(isinstance(g, str) for g in markers)


class TestRunSingleSample:
    """Tests for run_single_sample function."""

    def test_returns_success_tuple(self, temp_dir):
        """Test successful run returns (sample_id, True, 'Success')."""
        from ezsp.segment_adata import run_single_sample
        
        with patch('ezsp.segment_adata.run_pipeline') as mock_pipeline:
            result = run_single_sample({
                "library_id": "test_sample",
                "log_dir": str(temp_dir),
            })
        
        assert result == ("test_sample", True, "Success")

    def test_returns_failure_tuple_on_error(self, temp_dir):
        """Test that errors return (sample_id, False, error_message)."""
        from ezsp.segment_adata import run_single_sample
        
        with patch('ezsp.segment_adata.run_pipeline', side_effect=ValueError("Test error")):
            result = run_single_sample({
                "library_id": "test_sample",
                "log_dir": str(temp_dir),
            })
        
        assert result[0] == "test_sample"
        assert result[1] is False
        assert "Test error" in result[2]

    def test_creates_crash_log_on_error(self, temp_dir):
        """Test that crash.log is created on error."""
        from ezsp.segment_adata import run_single_sample
        
        log_dir = temp_dir / "logs"
        
        with patch('ezsp.segment_adata.run_pipeline', side_effect=RuntimeError("Crash!")):
            run_single_sample({
                "library_id": "test_sample",
                "log_dir": str(log_dir),
            })
        
        crash_file = log_dir / "crash.log"
        assert crash_file.exists()
        content = crash_file.read_text()
        assert "Crash!" in content


class TestRunBatch:
    """Tests for run_batch function."""

    def test_runs_all_samples(self, temp_dir):
        """Test that all samples are processed."""
        from ezsp.segment_adata import run_batch
        
        batch_params = [
            {"library_id": "sample_1", "log_dir": str(temp_dir)},
            {"library_id": "sample_2", "log_dir": str(temp_dir)},
        ]
        
        with patch('ezsp.segment_adata.run_single_sample', return_value=("x", True, "")) as mock_run:
            run_batch(batch_params, max_workers=1)
        
        assert mock_run.call_count == 2


class TestMain:
    """Tests for main function."""

    def test_exits_without_config_arg(self, capsys):
        """Test that main exits if no config argument provided."""
        import sys
        from ezsp.segment_adata import main
        
        with patch.object(sys, 'argv', ['segment_adata.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        
        assert exc_info.value.code == 1

    def test_exits_if_config_not_found(self, capsys):
        """Test that main exits if config file doesn't exist."""
        import sys
        from ezsp.segment_adata import main
        
        with patch.object(sys, 'argv', ['segment_adata.py', '/nonexistent/config.yaml']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        
        assert exc_info.value.code == 1
