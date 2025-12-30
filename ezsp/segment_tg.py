#!/usr/bin/env python
"""
Tangram-based spatial deconvolution with watershed segmentation.

This module performs:
1. Image preprocessing and watershed segmentation
2. Cell count estimation via image features
3. Tangram spatial deconvolution with cell type annotation
4. Visualization and output generation

Usage:
    python -m src.segment_tg \
        --sdata-path /path/to/sdata.zarr \
        --sc-path /path/to/single_cell.h5ad \
        --output-dir /path/to/output \
        --library-id PLA4 \
        --image-key B05067G5_HE_regist
"""

import argparse
import logging
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import squidpy as sq
import tangram as tg
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from skimage.filters import threshold_multiotsu

from img_norm.imgnorms import normalize_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_for_squidpy(
    sdata: sd.SpatialData,
    image_key: str,
    library_id: str,
    table_key: str = "bin100_table",
    hires_scale: int = 0,
    lowres_scale: int = 4,
) -> AnnData:
    """
    Prepare an AnnData object with spatial uns metadata for Squidpy.
    
    This builds uns['spatial'] on-the-fly only when calling Squidpy,
    keeping the original sdata clean.
    
    Parameters
    ----------
    sdata : sd.SpatialData
        Input spatial data object.
    table_key : str
        Key for the table in sdata. Default: "bin100_table".
    image_key : str
        Key for the image in sdata. 
    library_id : str
        Library identifier for spatial metadata.
    hires_scale : int
        Scale level for high-resolution image. Default: 0.
    lowres_scale : int
        Scale level for low-resolution image. Default: 4.
    
    Returns
    -------
    AnnData
        AnnData object with spatial metadata in uns.
    """
    adata = sdata.tables[table_key].copy()
    
    # Load and normalize high-resolution image
    img_hires = sdata.images[image_key][f"scale{hires_scale}"]["image"]
    img_hires = normalize_image(img_hires, compute=True)
    img_hires = np.moveaxis(img_hires.values, 0, -1)
    
    # Load and normalize low-resolution image
    img_lowres = sdata.images[image_key][f"scale{lowres_scale}"]["image"]
    img_lowres = normalize_image(img_lowres, compute=True)
    img_lowres = np.moveaxis(img_lowres.values, 0, -1)
    
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": img_hires, "lowres": img_lowres},
            "scalefactors": {
                "tissue_hires_scalef": 1.0 / 2.0**hires_scale,
                "spot_diameter_fullres": 100,
                "tissue_lowres_scalef": 1.0 / 2.0**lowres_scale,
            },
        }
    }
    return adata


def compute_segmentation_thresholds(
    img: np.ndarray,
    n_classes: int = 3,
    output_dir: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """
    Compute multi-Otsu thresholds for image segmentation.
    
    Parameters
    ----------
    img : np.ndarray
        Input image array.
    n_classes : int
        Number of classes for multi-Otsu thresholding. Default: 3.
    output_dir : pathlib.Path, optional
        Directory to save histogram plot. If None, histogram is not saved.
    
    Returns
    -------
    np.ndarray
        Array of threshold values.
    """
    thresholds = threshold_multiotsu(np.array(img), classes=n_classes)
    logger.info(f"Computed multi-Otsu thresholds: {thresholds}")
    
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Computing histogram (this may take a moment for large images)...")
        start_time = time.time()
        
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(img).ravel(), bins=50, range=(0, 256), alpha=0.7)
        plt.vlines(thresholds, 0, plt.gca().get_ylim()[1], color="r", linewidth=2)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Image Histogram with Multi-Otsu Thresholds")
        plt.savefig(output_dir / "histogram.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "histogram.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Saved histogram to {output_dir / 'histogram.[pdf|png]'} ({elapsed:.1f}s)")
    
    return thresholds


def segment_and_extract_features(
    adata: AnnData,
    library_id: str,
    thresh: float,
    n_jobs: int = -1,
) -> AnnData:
    """
    Perform watershed segmentation and extract cell count features.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial images in uns.
    library_id : str
        Library identifier in uns['spatial'].
    thresh : float
        Threshold value for watershed segmentation.
    n_jobs : int
        Number of parallel jobs for feature calculation. Default: -1 (all CPUs).
    
    Returns
    -------
    AnnData
        AnnData object with cell_count in obs.
    """
    # Prevent numba from interfering with joblib parallelism
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    imgc = sq.im.ImageContainer(img=adata.uns["spatial"][library_id]["images"]["hires"])
    
    # Perform watershed segmentation
    sq.im.segment(img=imgc, layer="image", method="watershed", thresh=thresh, geq=False)
    logger.info("Completed watershed segmentation")
    
    # Calculate segmentation features
    features_kwargs = {
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "centroid"],
        }
    }
    sq.im.calculate_image_features(
        adata,
        imgc,
        layer="image",
        key_added="image_features",
        features_kwargs=features_kwargs,
        features="segmentation",
        mask_circle=True,
        n_jobs=n_jobs,
    )
    
    adata.obs["cell_count"] = adata.obsm["image_features"]["segmentation_label"]
    logger.info(f"Extracted cell counts. Total cells: {adata.obs['cell_count'].sum()}")
    
    return adata


def prepare_single_cell_reference(
    sc_path: str,
    n_top_genes: int = 100,
    groupby: str = "major_celltype",
) -> tuple[AnnData, list[str]]:
    """
    Load and prepare single-cell reference data for Tangram.
    
    Parameters
    ----------
    sc_path : str
        Path to single-cell h5ad file.
    n_top_genes : int
        Number of top marker genes per cell type. Default: 100.
    groupby : str
        Column in obs to group cells by. Default: "major_celltype".
    
    Returns
    -------
    tuple[AnnData, list[str]]
        Tuple of (prepared AnnData, list of marker genes).
    """
    adata_sc = sc.read_h5ad(sc_path)
    logger.info(f"Loaded single-cell reference: {adata_sc.shape}")
    
    # Rank genes using log-normalized data
    adata_sc.X = adata_sc.layers["log_norm"].copy()
    sc.tl.rank_genes_groups(adata_sc, groupby=groupby, use_raw=False)
    
    # Extract top marker genes
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:n_top_genes, :]
    marker_genes = np.unique(markers_df.melt().value.values).tolist()
    logger.info(f"Identified {len(marker_genes)} marker genes")
    
    # Reset to raw counts for Tangram
    adata_sc.X = adata_sc.layers["raw"].copy()
    
    return adata_sc, marker_genes


def run_tangram_deconvolution(
    adata_sc: AnnData,
    adata_sp: AnnData,
    genes: list[str],
    annotation: str = "major_celltype",
    num_epochs: int = 1000,
    device: str = "cpu",
    random_state: int = 404,
    min_cell_count: int = 1,
    min_gene_pct: float = 0.01,
) -> tuple[AnnData, AnnData]:
    """
    Run Tangram spatial deconvolution.
    
    Parameters
    ----------
    adata_sc : AnnData
        Single-cell reference AnnData.
    adata_sp : AnnData
        Spatial AnnData with cell_count in obs.
    genes : list[str]
        List of genes for mapping.
    annotation : str
        Cell type annotation column. Default: "major_celltype".
    num_epochs : int
        Number of training epochs. Default: 1000.
    device : str
        Device for computation ("cpu" or "cuda:N"). Default: "cpu".
    random_state : int
        Random seed for reproducibility. Default: 404.
    min_cell_count : int
        Minimum cell count per spot for filtering. Default: 1.
    min_gene_pct : float
        Minimum percentage of spots expressing a gene (0-1). Default: 0.01 (1%).
    
    Returns
    -------
    tuple[AnnData, AnnData]
        Tuple of (adata_sp with annotations, segmented adata).
    """
    # Find common genes
    genes_common = list(set(genes).intersection(set(adata_sp.var_names)))
    logger.info(f"Using {len(genes_common)} common genes for mapping")
    
    # Filter spots by cell count
    n_spots_before = adata_sp.n_obs
    adata_sp = adata_sp[adata_sp.obs["cell_count"] >= min_cell_count].copy()
    n_spots_after = adata_sp.n_obs
    logger.info(
        f"Filtered spots by cell_count >= {min_cell_count}: "
        f"{n_spots_before} -> {n_spots_after} spots ({n_spots_before - n_spots_after} removed)"
    )
    
    # Filter genes by percentage of spots expressing them
    n_genes_before = adata_sp.n_vars
    # Count spots where gene is expressed (count > 0)
    gene_expressed = (adata_sp.X > 0).sum(axis=0)
    if hasattr(gene_expressed, "A1"):  # sparse matrix
        gene_expressed = gene_expressed.A1
    else:
        gene_expressed = np.asarray(gene_expressed).flatten()
    pct_expressed = gene_expressed / adata_sp.n_obs
    genes_to_keep = adata_sp.var_names[pct_expressed >= min_gene_pct]
    adata_sp = adata_sp[:, genes_to_keep].copy()
    n_genes_after = adata_sp.n_vars
    logger.info(
        f"Filtered genes by >= {min_gene_pct*100:.1f}% spots expressing: "
        f"{n_genes_before} -> {n_genes_after} genes ({n_genes_before - n_genes_after} removed)"
    )
    
    # Update common genes after filtering
    genes_common = list(set(genes_common).intersection(set(adata_sp.var_names)))
    logger.info(f"After filtering: {len(genes_common)} common marker genes remaining")
    
    # Preprocess for Tangram
    tg.pp_adatas(adata_sc, adata_sp, genes=genes_common)
    
    # Map cells to space with density prior from cell counts
    density_prior = np.array(adata_sp.obs.cell_count) / adata_sp.obs.cell_count.sum()
    
    logger.info(
        f"Starting Tangram mapping: {adata_sc.n_obs} cells -> {adata_sp.n_obs} spots, "
        f"{num_epochs} epochs on {device}..."
    )
    start_time = time.time()
    
    ad_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode="constrained",
        target_count=adata_sp.obs.cell_count.sum(),
        density_prior=density_prior,
        num_epochs=num_epochs,
        device=device,
        random_state=random_state,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Completed cell-to-space mapping ({elapsed:.1f}s, {elapsed/num_epochs*1000:.1f}ms/epoch)")
    
    # Project cell type annotations
    tg.project_cell_annotations(ad_map, adata_sp, annotation=annotation)
    
    # Create segment-level cell type assignments
    tg.create_segment_cell_df(adata_sp)
    tg.count_cell_annotations(ad_map, adata_sc, adata_sp, annotation=annotation)
    
    # Deconvolve at segment level
    adata_segment = tg.deconvolve_cell_annotations(adata_sp)
    logger.info("Completed segment-level deconvolution")
    
    return adata_sp, adata_segment


def save_results(
    adata_sp: AnnData,
    adata_segment: AnnData,
    output_dir: pathlib.Path,
    celltypes: Optional[list[str]] = None,
    matplotlib_style: Optional[str] = None,
) -> None:
    """
    Save deconvolution results and visualizations.
    
    Parameters
    ----------
    adata_sp : AnnData
        Spatial AnnData with cell type projections.
    adata_segment : AnnData
        Segmented AnnData with cell type assignments.
    output_dir : pathlib.Path
        Output directory for results.
    celltypes : list[str], optional
        List of cell types to visualize. Default: common placenta cell types.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if celltypes is None:
        celltypes = ["STB", "CTB", "EVT", "Fibroblast", "Myofibroblast", "VEC", "Hofbauer cells"]
    
    # Filter to available cell types
    available_celltypes = [ct for ct in celltypes if ct in adata_sp.obs.columns]
    ## if given a matplotlib style file, plt.style.use that file
    if matplotlib_style is not None:
        plt.style.use(matplotlib_style)
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "#FF0022"], N=256)
    
    # Plot cell type projections
    if available_celltypes:
        fig = sq.pl.spatial_scatter(
            adata_sp,
            color=available_celltypes,
            shape=None,
            cmap=cmap,
            return_fig=True,
        )
        if fig is not None:
            fig.savefig(output_dir / "celltype_projections.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(output_dir / "celltype_projections.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved cell type projection plot")
    
    # Plot segment cluster assignments
    fig, ax = plt.subplots(1, 1, figsize=(7.09 / 4, 6.699 / 4))
    sq.pl.spatial_scatter(
        adata_segment,
        color="cluster",
        frameon=False,
        img_alpha=0.2,
        legend_fontsize=20,
        ax=ax,
    )
    fig.savefig(output_dir / "segment_tg.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "segment_tg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(1, 1, figsize=(7.09 / 4, 6.699 / 4))
    sq.pl.spatial_scatter(
        adata_segment,
        shape="square",
        color="cluster",
        frameon=False,
        img_alpha=0.2,
        legend_fontsize=20,
        ax=ax,
        img_res_key="lowres",
    )
    fig.savefig(output_dir / "segment_tg_he.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "segment_tg_he.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved segment plots")
    
    # Save h5ad files
    adata_sp.write_h5ad(output_dir / "sp_tg.h5ad")
    adata_segment.write_h5ad(output_dir / "segment.h5ad")
    logger.info(f"Saved results to {output_dir}")


def run_pipeline(
    sdata_path: str,
    sc_path: str,
    output_dir: str,
    library_id: str = "PLA4",
    image_key: str = "B05067G5_HE_regist",
    table_key: str = "bin100_table",
    hires_scale: int = 0,
    lowres_scale: int = 4,
    n_classes: int = 3,
    n_top_genes: int = 100,
    num_epochs: int = 1000,
    device: str = "cpu",
    n_jobs: int = -1,
    min_cell_count: int = 1,
    min_gene_pct: float = 0.01,
) -> None:
    """
    Run the complete Tangram segmentation and deconvolution pipeline.
    
    Parameters
    ----------
    sdata_path : str
        Path to input SpatialData zarr.
    sc_path : str
        Path to single-cell reference h5ad.
    output_dir : str
        Output directory for results.
    library_id : str
        Library identifier. Default: "PLA4".
    image_key : str
        Key for H&E image in sdata. Default: "B05067G5_HE_regist".
    table_key : str
        Key for table in sdata. Default: "bin100_table".
    hires_scale : int
        Scale level for high-res image. Default: 0.
    lowres_scale : int
        Scale level for low-res image. Default: 4.
    n_classes : int
        Number of Otsu threshold classes. Default: 3.
    n_top_genes : int
        Number of top marker genes per cell type. Default: 100.
    num_epochs : int
        Training epochs for Tangram. Default: 1000.
    device : str
        Computation device. Default: "cpu".
    n_jobs : int
        Parallel jobs for feature extraction. Default: -1.
    min_cell_count : int
        Minimum cell count per spot for filtering. Default: 1.
    min_gene_pct : float
        Minimum percentage of spots expressing a gene (0-1). Default: 0.01 (1%).
    """
    output_dir = pathlib.Path(output_dir)
    
    # Log version info
    logger.info(f"scanpy=={sc.__version__}")
    logger.info(f"squidpy=={sq.__version__}")
    logger.info(f"tangram=={tg.__version__}")
    
    # Load spatial data
    logger.info(f"Loading spatial data from {sdata_path}")
    sdata = sd.read_zarr(sdata_path)
    
    # Prepare for Squidpy
    adata = prepare_for_squidpy(
        sdata,
        table_key=table_key,
        image_key=image_key,
        library_id=library_id,
        hires_scale=hires_scale,
        lowres_scale=lowres_scale,
    )
    
    # Compute segmentation thresholds
    imgc = sq.im.ImageContainer(img=adata.uns["spatial"][library_id]["images"]["hires"])
    img = imgc["image"]
    thresholds = compute_segmentation_thresholds(img, n_classes=n_classes, output_dir=output_dir)
    
    # Segment and extract features
    adata = segment_and_extract_features(
        adata,
        library_id=library_id,
        thresh=thresholds[1],  # Use middle threshold
        n_jobs=n_jobs,
    )
    
    # Prepare single-cell reference
    adata_sc, marker_genes = prepare_single_cell_reference(sc_path, n_top_genes=n_top_genes)
    
    # Run Tangram deconvolution
    adata_sp, adata_segment = run_tangram_deconvolution(
        adata_sc,
        adata,
        genes=marker_genes,
        num_epochs=num_epochs,
        device=device,
        min_cell_count=min_cell_count,
        min_gene_pct=min_gene_pct,
    )
    
    # Save results
    save_results(adata_sp, adata_segment, output_dir)
    
    logger.info("Pipeline completed successfully")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, optionally loading defaults from a YAML config file."""
    parser = argparse.ArgumentParser(
        description="Tangram-based spatial deconvolution with watershed segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI arguments override config file values.",
    )
    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate an example config file at the specified path and exit.",
    )
    
    # Required arguments (can be provided via config)
    parser.add_argument(
        "--sdata-path",
        type=str,
        default=None,
        help="Path to input SpatialData zarr",
    )
    parser.add_argument(
        "--sc-path",
        type=str,
        default=None,
        help="Path to single-cell reference h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    
    # Optional arguments
    parser.add_argument(
        "--library-id",
        type=str,
        default=None,
        help="Library identifier for spatial metadata",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Key for H&E image in sdata",
    )
    parser.add_argument(
        "--table-key",
        type=str,
        default="bin100_table",
        help="Key for table in sdata",
    )
    parser.add_argument(
        "--hires-scale",
        type=int,
        default=0,
        help="Scale level for high-resolution image",
    )
    parser.add_argument(
        "--lowres-scale",
        type=int,
        default=4,
        help="Scale level for low-resolution image",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=3,
        help="Number of Otsu threshold classes",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=100,
        help="Number of top marker genes per cell type",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Training epochs for Tangram",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (cpu or cuda:N)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for feature extraction (-1 for all CPUs)",
    )
    parser.add_argument(
        "--min-cell-count",
        type=int,
        default=1,
        help="Minimum cell count per spot for filtering",
    )
    parser.add_argument(
        "--min-gene-pct",
        type=float,
        default=0.01,
        help="Minimum percentage of spots expressing a gene (0-1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max parallel workers for batch mode (1 = sequential)",
    )
    
    args = parser.parse_args()
    
    # Handle --generate-config
    if args.generate_config:
        generate_example_config(args.generate_config)
        parser.exit(0)
    
    # Load config file if specified
    if args.config:
        args = load_config_and_merge(args, parser)
    
    # Validate required arguments
    required = ["sdata_path", "sc_path", "output_dir", "library_id", "image_key"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}. Provide via CLI or config file.")
    
    return args


def load_config_and_merge(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Load YAML config and merge with CLI arguments.
    CLI arguments take precedence over config file values.
    """
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    logger.info(f"Loaded config from {config_path}")
    
    # Map YAML keys (with underscores) to argparse attribute names
    # YAML uses underscores, argparse converts hyphens to underscores
    for key, value in config.items():
        attr_name = key.replace("-", "_")
        # Only set from config if CLI didn't provide a value (is None or default)
        current_value = getattr(args, attr_name, None)
        if current_value is None:
            setattr(args, attr_name, value)
    
    return args


def generate_example_config(output_path: str) -> None:
    """Generate an example YAML config file."""
    example_config = """\
# Tangram Segmentation Pipeline Configuration
# All paths can be absolute or relative to the working directory

# ============================================================
# BATCH MODE: Use lists to process multiple samples in parallel
# ============================================================
# sdata_path:
#   - /path/to/sample1.zarr
#   - /path/to/sample2.zarr
# sc_path:
#   - /path/to/ref1.h5ad
#   - /path/to/ref2.h5ad
# output_dir:
#   - /path/to/output1
#   - /path/to/output2
# library_id:
#   - PLA4
#   - PLA7
# image_key:
#   - image_key_1
#   - image_key_2
# max_workers: 2  # Number of parallel workers

# ============================================================
# SINGLE MODE: Use single values for one sample
# ============================================================
sdata_path: /path/to/sdata.zarr
sc_path: /path/to/single_cell.h5ad
output_dir: /path/to/output
library_id: PLA4
image_key: B05067G5_HE_regist

# Optional parameters (defaults shown)
table_key: bin100_table
hires_scale: 0
lowres_scale: 4
n_classes: 3
n_top_genes: 100
num_epochs: 1000
device: cpu  # or cuda:0 for GPU
n_jobs: -1  # -1 for all CPUs

# Filtering parameters
min_cell_count: 1  # Minimum cells per spot
min_gene_pct: 0.01  # Minimum % of spots expressing gene (0.01 = 1%)
"""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(example_config)
    print(f"Generated example config at: {output_path}")


def run_single_sample(kwargs: dict) -> tuple[str, bool, str]:
    """
    Run pipeline for a single sample. Used by ProcessPoolExecutor.
    
    Returns tuple of (sample_id, success, message).
    """
    sample_id = kwargs.get("library_id", "unknown")
    try:
        run_pipeline(**kwargs)
        return (sample_id, True, "Success")
    except Exception as e:
        return (sample_id, False, str(e))


def run_batch(
    batch_params: list[dict],
    max_workers: int = 1,
) -> None:
    """
    Run pipeline for multiple samples in parallel.
    
    Parameters
    ----------
    batch_params : list[dict]
        List of parameter dicts, one per sample.
    max_workers : int
        Maximum number of parallel workers. Default: 1 (sequential).
    """
    n_samples = len(batch_params)
    logger.info(f"Starting batch processing: {n_samples} samples with {max_workers} workers")
    
    start_time = time.time()
    results = []
    
    if max_workers == 1:
        # Sequential processing
        for i, params in enumerate(batch_params, 1):
            sample_id = params.get("library_id", f"sample_{i}")
            logger.info(f"[{i}/{n_samples}] Processing {sample_id}...")
            result = run_single_sample(params)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(run_single_sample, params): params.get("library_id", f"sample_{i}")
                for i, params in enumerate(batch_params, 1)
            }
            
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result[1] else "✗"
                    logger.info(f"[{len(results)}/{n_samples}] {status} {sample_id}")
                except Exception as e:
                    results.append((sample_id, False, str(e)))
                    logger.error(f"[{len(results)}/{n_samples}] ✗ {sample_id}: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    successes = sum(1 for _, success, _ in results if success)
    failures = n_samples - successes
    
    logger.info(f"\nBatch completed in {elapsed:.1f}s")
    logger.info(f"  Successes: {successes}/{n_samples}")
    if failures > 0:
        logger.warning(f"  Failures: {failures}/{n_samples}")
        for sample_id, success, msg in results:
            if not success:
                logger.warning(f"    - {sample_id}: {msg}")


if __name__ == "__main__":
    args = parse_args()
    
    # Check if batch mode (list parameters)
    batch_keys = ["sdata_path", "sc_path", "output_dir", "library_id", "image_key"]
    is_batch = isinstance(getattr(args, batch_keys[0]), list)
    
    if is_batch:
        # Validate all batch keys are lists of same length
        lengths = {k: len(getattr(args, k)) for k in batch_keys if isinstance(getattr(args, k), list)}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Batch parameters must have same length: {lengths}")
        
        n_samples = list(lengths.values())[0]
        
        # Build list of param dicts for each sample
        batch_params = []
        for i in range(n_samples):
            params = {
                "sdata_path": args.sdata_path[i],
                "sc_path": args.sc_path[i],
                "output_dir": args.output_dir[i],
                "library_id": args.library_id[i],
                "image_key": args.image_key[i],
                "table_key": args.table_key,
                "hires_scale": args.hires_scale,
                "lowres_scale": args.lowres_scale,
                "n_classes": args.n_classes,
                "n_top_genes": args.n_top_genes,
                "num_epochs": args.num_epochs,
                "device": args.device,
                "n_jobs": args.n_jobs,
                "min_cell_count": args.min_cell_count,
                "min_gene_pct": args.min_gene_pct,
            }
            batch_params.append(params)
        
        run_batch(batch_params, max_workers=args.max_workers)
    else:
        # Single sample mode
        run_pipeline(
            sdata_path=args.sdata_path,
            sc_path=args.sc_path,
            output_dir=args.output_dir,
            library_id=args.library_id,
            image_key=args.image_key,
            table_key=args.table_key,
            hires_scale=args.hires_scale,
            lowres_scale=args.lowres_scale,
            n_classes=args.n_classes,
            n_top_genes=args.n_top_genes,
            num_epochs=args.num_epochs,
            device=args.device,
            n_jobs=args.n_jobs,
            min_cell_count=args.min_cell_count,
            min_gene_pct=args.min_gene_pct,
        )
