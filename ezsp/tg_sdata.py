import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel, Image2DModel
import squidpy as sq
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData

from ._img_utils import normalize_image

def segment4sdata(
    sdata: sd.SpatialData,
    scale: int = 0,
    save: str = "",
    image_key: str = "stain",
    points_key: str = "bin1_genes",  # Stereo-seq bin1 points
) -> sd.SpatialData:
    """Watershed segmentation and aggregate Stereo-seq bin1 to cells."""
    import os
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    # 1. Segment from HE/ssDNA image
    img_da = sdata.images[image_key][f"scale{scale}"]["image"]
    
    # Get lazy normalized image for storage, and computed array for processing
    img_norm_lazy = normalize_image(img_da, compute=False)  # Lazy dask-backed DataArray
    img_norm_computed = img_norm_lazy.compute().values      # Numpy array for segmentation
    
    # Store normalized image as proper SpatialData image (dask-backed DataTree)
    original_transforms = sd.transformations.get_transformation(img_da, get_all=True)
    sdata.images[f"{image_key}_norm"] = Image2DModel.parse(
        img_norm_lazy.data,  # dask array
        dims=img_da.dims,
        transformations=original_transforms,
    )
    
    from .segment import compute_segmentation_thresholds, segment
    imgc = sq.im.ImageContainer(img=img_norm_computed)
    thresholds = compute_segmentation_thresholds(imgc["image"], n_classes=3, output_dir=None)
    imgc = segment(img=img_norm_computed, thresh=thresholds[1])

    # 2. Extract mask and add as labels
    seg_mask = imgc["segmented_watershed"].values.squeeze()
    from spatialdata.models import Labels2DModel
    sdata.labels["cell_labels"] = Labels2DModel.parse(
        seg_mask, 
        transformations=original_transforms,
    )
    
    # 3. Convert labels to polygons
    cell_shapes = sd.to_polygons(sdata.labels["cell_labels"])
    sdata.shapes["cells"] = cell_shapes
    
    # 4. Assign points to cells using labels
    # Stereo-seq points don't have gene column - use labels to assign points to cells
    points_df = sdata.points[points_key].compute()
    
    # Get cell label for each point by looking up in the segmentation mask
    # Coordinates need to be integers for indexing
    y_coords = points_df["y"].values.astype(int)
    x_coords = points_df["x"].values.astype(int)
    
    # Clip to image bounds
    y_coords = np.clip(y_coords, 0, seg_mask.shape[0] - 1)
    x_coords = np.clip(x_coords, 0, seg_mask.shape[1] - 1)
    
    # Assign cell labels to points
    cell_ids = seg_mask[y_coords, x_coords]
    points_df["cell_id"] = cell_ids
    
    # Filter out points not in any cell (label 0 = background)
    points_in_cells = points_df[points_df["cell_id"] > 0]
    
    # Count points per cell
    cell_counts = points_in_cells.groupby("cell_id").size().reset_index(name="n_transcripts")
    
    # Store assignment info in sdata for downstream use
    sdata.attrs["points_to_cells"] = {
        "points_key": points_key,
        "n_points_total": len(points_df),
        "n_points_assigned": len(points_in_cells),
        "n_cells_with_points": len(cell_counts),
    }
    
    if save:
        from . import _io
        _io.safe_update_sdata(sdata, save)
    return sdata


def create_cell_gene_matrix(
    sdata: sd.SpatialData,
    points_key: str = "bin1_genes",
    gene_column: str = "gene",
    labels_key: str = "cell_labels",
    shapes_key: str = "cells",
    min_counts: int = 0,
) -> AnnData:
    """
    Create a cell-by-gene count matrix from transcript points assigned to cells.
    
    Parameters
    ----------
    sdata : sd.SpatialData
        SpatialData object with segmented cells and transcript points.
    points_key : str
        Key for the points element containing transcripts.
    gene_column : str
        Column name in points dataframe containing gene names.
    labels_key : str
        Key for the labels element containing cell segmentation mask.
    shapes_key : str
        Key for the shapes element containing cell polygons.
    min_counts : int
        Minimum number of transcripts per cell to include.
        
    Returns
    -------
    AnnData
        Cell-by-gene count matrix with:
        - .X: sparse count matrix (cells × genes)
        - .obs: cell metadata (cell_id, centroid_x, centroid_y, n_transcripts, area)
        - .var: gene metadata
        - .obsm['spatial']: cell centroid coordinates for plotting
    """
    # Get segmentation mask
    seg_mask = sdata.labels[labels_key].values
    if seg_mask.ndim > 2:
        seg_mask = seg_mask.squeeze()
    
    # Get points and assign to cells
    points_df = sdata.points[points_key].compute()
    
    # Check if gene column exists
    if gene_column not in points_df.columns:
        raise ValueError(
            f"Gene column '{gene_column}' not found in points. "
            f"Available columns: {list(points_df.columns)}"
        )
    
    # Get cell label for each point
    y_coords = points_df["y"].values.astype(int)
    x_coords = points_df["x"].values.astype(int)
    
    # Clip to image bounds
    y_coords = np.clip(y_coords, 0, seg_mask.shape[0] - 1)
    x_coords = np.clip(x_coords, 0, seg_mask.shape[1] - 1)
    
    # Assign cell labels
    points_df["cell_id"] = seg_mask[y_coords, x_coords]
    
    # Filter out background (cell_id == 0)
    points_in_cells = points_df[points_df["cell_id"] > 0].copy()
    
    # Get unique cells and genes
    unique_cells = np.sort(points_in_cells["cell_id"].unique())
    unique_genes = np.sort(points_in_cells[gene_column].unique())
    
    # Create cell and gene index mappings
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}
    
    # Count transcripts per cell-gene pair
    counts = points_in_cells.groupby(["cell_id", gene_column]).size().reset_index(name="count")
    
    # Build sparse matrix
    row_indices = counts["cell_id"].map(cell_to_idx).values
    col_indices = counts[gene_column].map(gene_to_idx).values
    data = counts["count"].values
    
    count_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_cells), len(unique_genes)),
        dtype=np.float32,
    )
    
    # Build obs (cell metadata)
    # Calculate per-cell statistics
    cell_stats = points_in_cells.groupby("cell_id").agg(
        n_transcripts=("cell_id", "size"),
        centroid_x=("x", "mean"),
        centroid_y=("y", "mean"),
    ).reset_index()
    
    # Add area from shapes if available
    if shapes_key in sdata.shapes:
        shapes = sdata.shapes[shapes_key]
        cell_stats["area"] = cell_stats["cell_id"].map(
            dict(zip(shapes.index, shapes.geometry.area))
        )
    
    # Filter by min_counts
    if min_counts > 0:
        valid_cells = cell_stats[cell_stats["n_transcripts"] >= min_counts]["cell_id"].values
        valid_mask = np.isin(unique_cells, valid_cells)
        unique_cells = unique_cells[valid_mask]
        count_matrix = count_matrix[valid_mask, :]
        cell_stats = cell_stats[cell_stats["cell_id"].isin(unique_cells)]
    
    # Create obs DataFrame
    obs = pd.DataFrame({
        "cell_id": unique_cells,
    })
    obs = obs.merge(cell_stats, on="cell_id", how="left")
    obs.index = [f"cell_{cid}" for cid in obs["cell_id"]]
    
    # Create var DataFrame
    var = pd.DataFrame(index=unique_genes)
    var.index.name = "gene"
    
    # Build AnnData
    adata = AnnData(
        X=count_matrix,
        obs=obs,
        var=var,
    )
    
    # Add spatial coordinates to obsm
    adata.obsm["spatial"] = obs[["centroid_x", "centroid_y"]].values
    
    # Store metadata
    adata.uns["spatial_info"] = {
        "points_key": points_key,
        "gene_column": gene_column,
        "n_cells": len(unique_cells),
        "n_genes": len(unique_genes),
        "n_transcripts_total": int(count_matrix.sum()),
    }
    
    return adata
