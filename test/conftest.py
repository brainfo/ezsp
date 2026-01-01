"""
Shared fixtures for ezsp tests.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def sample_image():
    """Create a simple test image with distinct regions for segmentation testing."""
    # Create 100x100 grayscale image with 3 distinct intensity regions
    img = np.zeros((100, 100), dtype=np.uint8)
    img[0:33, :] = 50      # Low intensity region
    img[33:66, :] = 128    # Medium intensity region
    img[66:100, :] = 220   # High intensity region
    return img


@pytest.fixture
def sample_rgb_image():
    """Create a simple RGB test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:50, :, 0] = 200   # Red in top half
    img[50:100, :, 2] = 200  # Blue in bottom half
    return img


@pytest.fixture
def mock_adata():
    """Create a minimal AnnData object for testing."""
    n_obs = 50
    n_vars = 100
    
    # Create random count matrix
    X = np.random.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create obs dataframe with coordinates
    obs = pd.DataFrame({
        'in_tissue': [1] * n_obs,
        'array_row': np.arange(n_obs) // 10,
        'array_col': np.arange(n_obs) % 10,
    }, index=[f'spot_{i}' for i in range(n_obs)])
    
    # Create var dataframe
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars)])
    
    # Create AnnData
    adata = AnnData(X=X, obs=obs, var=var)
    
    return adata


@pytest.fixture
def mock_adata_with_spatial(mock_adata, sample_rgb_image):
    """Create AnnData with spatial metadata for Squidpy compatibility."""
    adata = mock_adata.copy()
    library_id = "test_sample"
    
    adata.uns["spatial"] = {
        library_id: {
            "images": {
                "hires": sample_rgb_image,
                "lowres": sample_rgb_image[::4, ::4],  # Downscaled
            },
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "spot_diameter_fullres": 100,
                "tissue_lowres_scalef": 0.25,
            },
        }
    }
    
    # Add spatial coordinates in obsm
    adata.obsm["spatial"] = np.stack([
        adata.obs["array_row"].values * 10,
        adata.obs["array_col"].values * 10,
    ], axis=1).astype(np.float64)
    
    return adata


@pytest.fixture
def mock_sc_adata():
    """Create a mock single-cell AnnData for Tangram testing."""
    n_cells = 200
    n_genes = 100
    
    # Create count matrix
    X = np.random.poisson(10, size=(n_cells, n_genes)).astype(np.float32)
    
    # Create cell type annotations - 4 cell types
    cell_types = ['TypeA', 'TypeB', 'TypeC', 'TypeD']
    obs = pd.DataFrame({
        'major_celltype': np.random.choice(cell_types, n_cells),
    }, index=[f'cell_{i}' for i in range(n_cells)])
    
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    
    # Add layers required by prepare_single_cell_reference
    adata.layers['raw'] = X.copy()
    adata.layers['log_norm'] = np.log1p(X)
    
    return adata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'sdata_path': '/path/to/sdata.zarr',
        'sc_path': '/path/to/sc_reference.h5ad',
        'output_dir': '/path/to/output',
        'log_dir': '/path/to/logs',
        'library_id': 'test_sample',
        'image_key': 'stain',
        'table_key': 'bin100_table',
        'hires_scale': 0,
        'lowres_scale': 4,
        'n_classes': 3,
        'n_top_genes': 100,
        'num_epochs': 10,  # Small for testing
        'device': 'cpu',
        'n_jobs': 1,
        'min_cells_per_type': 5,
        'min_cell_count': 1,
    }


@pytest.fixture
def sample_batch_config():
    """Create a sample batch configuration dictionary."""
    return {
        'sdata_path': ['/path/to/sdata1.zarr', '/path/to/sdata2.zarr'],
        'sc_path': ['/path/to/sc1.h5ad', '/path/to/sc2.h5ad'],
        'output_dir': ['/path/to/output1', '/path/to/output2'],
        'log_dir': ['/path/to/logs1', '/path/to/logs2'],
        'library_id': ['sample_1', 'sample_2'],
        'image_key': ['stain', 'stain'],
        'max_workers': 1,
        'num_epochs': 10,
    }
