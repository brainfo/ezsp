#!/usr/bin/env python
"""
Tangram-based spatial deconvolution with watershed segmentation.

Usage:
    python -m src.segment_tg config/tg.yaml
"""

import logging
import os
import pathlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import yaml
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

from .img_utils import normalize_image

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> pathlib.Path:
    """Configure logging to file and minimal console output."""
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tangram_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear existing handlers
    logging.root.handlers = []
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    ))
    
    # Console handler (minimal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)
    print(f"Logging to: {log_file}")
    return log_file


def prepare_for_squidpy(
    sdata: sd.SpatialData,
    image_key: str,
    library_id: str,
    table_key: str = "bin100_table",
    hires_scale: int = 0,
    lowres_scale: int = 4,
) -> AnnData:
    """Prepare AnnData with spatial metadata for Squidpy."""
    adata = sdata.tables[table_key].copy()
    
    img_hires = sdata.images[image_key][f"scale{hires_scale}"]["image"]
    img_hires = img_utils.normalize_image(img_hires, compute=True)
    img_hires = np.moveaxis(img_hires.values, 0, -1)
    
    img_lowres = sdata.images[image_key][f"scale{lowres_scale}"]["image"]
    img_lowres = img_utils.normalize_image(img_lowres, compute=True)
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


def segment_and_extract_features(
    adata: AnnData,
    library_id: str,
    thresh: float,
    n_jobs: int = -1,
) -> AnnData:
    """Perform watershed segmentation and extract cell count features."""
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    imgc = sq.im.ImageContainer(img=adata.uns["spatial"][library_id]["images"]["hires"])
    sq.im.segment(img=imgc, layer="image", method="watershed", thresh=thresh, geq=False)
    logger.info("Completed watershed segmentation")
    
    features_kwargs = {
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "centroid"],
        }
    }
    sq.im.calculate_image_features(
        adata, imgc, layer="image", key_added="image_features",
        features_kwargs=features_kwargs, features="segmentation",
        mask_circle=True, n_jobs=n_jobs,
    )
    
    adata.obs["cell_count"] = adata.obsm["image_features"]["segmentation_label"]
    logger.info(f"Extracted cell counts. Total: {adata.obs['cell_count'].sum()}")
    return adata


def prepare_single_cell_reference(
    sc_path: str,
    n_top_genes: int = 100,
    groupby: str = "major_celltype",
    min_cells_per_type: int = 10,
) -> tuple[AnnData, list[str]]:
    """Load and prepare single-cell reference data for Tangram."""
    adata_sc = sc.read_h5ad(sc_path)
    logger.info(f"Loaded single-cell reference: {adata_sc.shape}")
    
    # Filter cell types by minimum cell count
    celltype_counts = adata_sc.obs[groupby].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cells_per_type].index.tolist()
    removed = celltype_counts[celltype_counts < min_cells_per_type]
    
    if len(removed) > 0:
        logger.warning(f"Filtering {len(removed)} cell types with < {min_cells_per_type} cells: {dict(removed)}")
    
    adata_sc = adata_sc[adata_sc.obs[groupby].isin(valid_celltypes)].copy()
    logger.info(f"After filtering: {adata_sc.shape}, {len(valid_celltypes)} cell types")
    
    # Rank genes
    adata_sc.X = adata_sc.layers["log_norm"].copy()
    sc.tl.rank_genes_groups(adata_sc, groupby=groupby, use_raw=False)
    
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:n_top_genes, :]
    marker_genes = np.unique(markers_df.melt().value.values).tolist()
    logger.info(f"Identified {len(marker_genes)} marker genes")
    
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
    output_dir: Optional[pathlib.Path] = None,
) -> tuple[AnnData, AnnData]:
    """Run Tangram spatial deconvolution."""
    genes_common = list(set(genes).intersection(set(adata_sp.var_names)))
    logger.info(f"Using {len(genes_common)} common genes for mapping")
    
    tg.pp_adatas(adata_sc, adata_sp, genes=genes_common)
    
    density_prior = np.array(adata_sp.obs.cell_count) / adata_sp.obs.cell_count.sum()
    
    logger.info(f"Starting Tangram: {adata_sc.n_obs} cells -> {adata_sp.n_obs} spots, {num_epochs} epochs")
    start_time = time.time()
    
    ad_map = tg.map_cells_to_space(
        adata_sc, adata_sp,
        mode="constrained",
        target_count=adata_sp.obs.cell_count.sum(),
        density_prior=density_prior,
        num_epochs=num_epochs,
        device=device,
        random_state=random_state,
    )
    
    logger.info(f"Completed mapping ({time.time() - start_time:.1f}s)")
    
    # Project cell type annotations
    tg.project_cell_annotations(ad_map, adata_sp, annotation=annotation)
    ct_names = adata_sp.uns["tangram_ct_pred_names"]
    ct_pred = adata_sp.obsm["tangram_ct_pred"]
    for i, ct in enumerate(ct_names):
        adata_sp.obs[ct] = ct_pred[:, i]
    adata_sp.write_h5ad(output_dir / "sp_tg_map.h5ad")
    # Segment-level deconvolution
    tg.create_segment_cell_df(adata_sp)
    tg.count_cell_annotations(ad_map, adata_sc, adata_sp, annotation=annotation)
    adata_segment = tg.deconvolve_cell_annotations(adata_sp)
    logger.info("Completed segment-level deconvolution")
    
    return adata_sp, adata_segment


def save_results(
    adata_sp: AnnData,
    adata_segment: AnnData,
    output_dir: pathlib.Path,
    celltypes: Optional[list[str]] = None,
) -> None:
    """Save deconvolution results and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save h5ad files
    adata_sp.write_h5ad(output_dir / "sp_tg.h5ad")
    adata_segment.write_h5ad(output_dir / "segment.h5ad")
    logger.info(f"Saved results to {output_dir}")
    
    if celltypes is None:
        celltypes = ["STB", "CTB", "EVT", "Fibroblast", "Myofibroblast", "VEC", "Hofbauer cells"]
    
    available = [ct for ct in celltypes if ct in adata_sp.obs.columns]
    cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "#FF0022"], N=256)
    
    # Plot cell type projections
    if available:
        n_cols = min(4, len(available))
        n_rows = (len(available) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, ct in enumerate(available):
            sq.pl.spatial_scatter(adata_sp, color=ct, shape=None, cmap=cmap, ax=axes[i])
            axes[i].set_title(ct)
        
        # Hide unused axes
        for j in range(len(available), len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        fig.savefig(output_dir / "celltype_projections.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "celltype_projections.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cell type projection plot")
    
    # Plot segment cluster assignments
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sq.pl.spatial_scatter(adata_segment, color="cluster", frameon=False, ax=ax)
    fig.savefig(output_dir / "segment_tg.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "segment_tg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sq.pl.spatial_scatter(adata_segment, color="cluster", frameon=False, ax=ax, img_res_key="lowres")
    fig.savefig(output_dir / "segment_tg_he.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "segment_tg_he.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved segment plots")


def run_pipeline(
    sdata_path: str,
    sc_path: str,
    output_dir: str,
    log_dir: str,
    library_id: str,
    image_key: str,
    table_key: str = "bin100_table",
    hires_scale: int = 0,
    lowres_scale: int = 4,
    n_classes: int = 3,
    n_top_genes: int = 100,
    num_epochs: int = 1000,
    device: str = "cpu",
    n_jobs: int = -1,
    min_cells_per_type: int = 10,
    min_cell_count: int = 1,
) -> None:
    """Run the complete Tangram pipeline."""
    setup_logging(log_dir)
    
    logger.info(f"=== Starting {library_id} ===")
    logger.info(f"scanpy=={sc.__version__}, squidpy=={sq.__version__}, tangram=={tg.__version__}")
    
    # Load spatial data
    logger.info(f"Loading spatial data from {sdata_path}")
    output_dir = pathlib.Path(output_dir)
    sdata = sd.read_zarr(sdata_path)
    
    adata = prepare_for_squidpy(
        sdata, table_key=table_key, image_key=image_key,
        library_id=library_id, hires_scale=hires_scale, lowres_scale=lowres_scale,
    )
    
    # Segmentation and feature extraction
    from .segment import compute_segmentation_thresholds
    imgc = sq.im.ImageContainer(img=adata.uns["spatial"][library_id]["images"]["hires"])
    thresholds = compute_segmentation_thresholds(imgc["image"], n_classes=n_classes, output_dir=output_dir)
    adata = segment_and_extract_features(adata, library_id=library_id, thresh=thresholds[1], n_jobs=n_jobs)
    
    # Filter early (before tangram to keep indices consistent)
    if min_cell_count > 0:
        n_before = adata.n_obs
        adata = adata[adata.obs["cell_count"] >= min_cell_count].copy()
        logger.info(f"Filtered spots by cell_count >= {min_cell_count}: {n_before} -> {adata.n_obs}")
    
    # Single-cell reference
    adata_sc, marker_genes = prepare_single_cell_reference(
        sc_path, n_top_genes=n_top_genes, min_cells_per_type=min_cells_per_type,
    )
    
    # Tangram deconvolution
    adata_sp, adata_segment = run_tangram_deconvolution(
        adata_sc, adata, genes=marker_genes, num_epochs=num_epochs, device=device, output_dir=output_dir,
    )
    
    save_results(adata_sp, adata_segment, output_dir)
    logger.info(f"=== Completed {library_id} ===")


def run_single_sample(kwargs: dict) -> tuple[str, bool, str]:
    """Run pipeline for a single sample."""
    sample_id = kwargs.get("library_id", "unknown")
    try:
        run_pipeline(**kwargs)
        return (sample_id, True, "Success")
    except Exception as e:
        # Log to sample-specific crash file
        log_dir = pathlib.Path(kwargs.get("log_dir", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        crash_file = log_dir / "crash.log"
        with open(crash_file, "w") as f:
            f.write(f"Crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {e}\n\n{traceback.format_exc()}")
        return (sample_id, False, str(e))


def run_batch(batch_params: list[dict], max_workers: int = 1) -> None:
    """Run pipeline for multiple samples."""
    n_samples = len(batch_params)
    print(f"Starting batch: {n_samples} samples, {max_workers} workers")
    
    start_time = time.time()
    results = []
    
    if max_workers == 1:
        for i, params in enumerate(batch_params, 1):
            sample_id = params.get("library_id", f"sample_{i}")
            print(f"[{i}/{n_samples}] Processing {sample_id}...")
            results.append(run_single_sample(params))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_sample, p): p.get("library_id") for p in batch_params}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "✓" if result[1] else "✗"
                print(f"[{len(results)}/{n_samples}] {status} {result[0]}")
    
    # Summary
    successes = sum(1 for _, ok, _ in results if ok)
    print(f"\nCompleted in {time.time() - start_time:.1f}s: {successes}/{n_samples} succeeded")
    for sample_id, ok, msg in results:
        if not ok:
            print(f"  FAILED {sample_id}: {msg}")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.segment_tg config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not pathlib.Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Check batch mode
    batch_keys = ["sdata_path", "sc_path", "output_dir", "library_id", "image_key", "log_dir"]
    is_batch = isinstance(config.get("sdata_path"), list)
    
    if is_batch:
        lengths = {k: len(config[k]) for k in batch_keys if isinstance(config.get(k), list)}
        if len(set(lengths.values())) > 1:
            print(f"ERROR: Batch parameters must have same length: {lengths}")
            sys.exit(1)
        
        n_samples = list(lengths.values())[0]
        batch_params = []
        for i in range(n_samples):
            params = {
                "sdata_path": config["sdata_path"][i],
                "sc_path": config["sc_path"][i],
                "output_dir": config["output_dir"][i],
                "log_dir": config["log_dir"][i],
                "library_id": config["library_id"][i],
                "image_key": config["image_key"][i],
                "table_key": config.get("table_key", "bin100_table"),
                "hires_scale": config.get("hires_scale", 0),
                "lowres_scale": config.get("lowres_scale", 4),
                "n_classes": config.get("n_classes", 3),
                "n_top_genes": config.get("n_top_genes", 100),
                "num_epochs": config.get("num_epochs", 1000),
                "device": config.get("device", "cpu"),
                "n_jobs": config.get("n_jobs", -1),
                "min_cells_per_type": config.get("min_cells_per_type", 10),
                "min_cell_count": config.get("min_cell_count", 1),
            }
            batch_params.append(params)
        
        run_batch(batch_params, max_workers=config.get("max_workers", 1))
    else:
        run_pipeline(
            sdata_path=config["sdata_path"],
            sc_path=config["sc_path"],
            output_dir=config["output_dir"],
            log_dir=config["log_dir"],
            library_id=config["library_id"],
            image_key=config["image_key"],
            table_key=config.get("table_key", "bin100_table"),
            hires_scale=config.get("hires_scale", 0),
            lowres_scale=config.get("lowres_scale", 4),
            n_classes=config.get("n_classes", 3),
            n_top_genes=config.get("n_top_genes", 100),
            num_epochs=config.get("num_epochs", 1000),
            device=config.get("device", "cpu"),
            n_jobs=config.get("n_jobs", -1),
            min_cells_per_type=config.get("min_cells_per_type", 10),
            min_cell_count=config.get("min_cell_count", 1),
        )


if __name__ == "__main__":
    main()