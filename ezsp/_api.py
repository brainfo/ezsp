"""API module - High-level pipeline and batch execution functions."""
import logging
import pathlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import scanpy as sc
import spatialdata as sd
import squidpy as sq
import tangram as tg

from ._img_utils import compute_segmentation_thresholds
from ._pp import prepare_for_squidpy, prepare_single_cell_reference, setup_logging
from .tg_adata import run_tangram_deconvolution, save_results, segment_and_extract_features
from .pl import load_adata_sp, load_mapped_ge, plot_genes_for_celltype, plot_tangram_proportion

logger = logging.getLogger(__name__)


# ============================================================================
# TANGRAM PIPELINE
# ============================================================================

def run_tangram_pipeline(
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
    cell_types: set[str] | None = None,
    cell_types_proportion: dict[str, float] | None = None,
    min_segment_confidence: float = 0.00,
    project_genes: bool = True,
    plot_genes: list[str] | None = None,
) -> None:
    """Run the complete Tangram pipeline."""
    setup_logging(log_dir)
    
    logger.info(f"=== Starting {library_id} ===")
    logger.info(f"scanpy=={sc.__version__}, squidpy=={sq.__version__}, tangram=={tg.__version__}")
    
    output_dir = pathlib.Path(output_dir)
    sdata = sd.read_zarr(sdata_path)
    
    adata = prepare_for_squidpy(
        sdata, table_key=table_key, image_key=image_key,
        library_id=library_id, hires_scale=hires_scale, lowres_scale=lowres_scale,
    )
    
    # Segmentation
    imgc = sq.im.ImageContainer(img=adata.uns["spatial"][library_id]["images"]["hires"])
    thresholds = compute_segmentation_thresholds(imgc["image"], n_classes=n_classes, output_dir=output_dir)
    adata, imgc = segment_and_extract_features(adata, library_id=library_id, thresh=thresholds[1], n_jobs=n_jobs)
    
    # Filter spots with low cell count
    if min_cell_count > 0:
        n_before = adata.n_obs
        adata = adata[adata.obs["cell_count"] >= min_cell_count].copy()
        logger.info(f"Filtered spots by cell_count >= {min_cell_count}: {n_before} -> {adata.n_obs}")
    
    # Single-cell reference
    if cell_types is None and cell_types_proportion is not None:
        cell_types = set(cell_types_proportion.keys())
    adata_sc, marker_genes = prepare_single_cell_reference(
        sc_path, n_top_genes=n_top_genes, min_cells_per_type=min_cells_per_type,
        cell_types=cell_types, cell_types_proportion=cell_types_proportion,
    )
    
    # Tangram deconvolution
    adata_sp, adata_segment, ad_ge = run_tangram_deconvolution(
        adata_sc, adata, genes=marker_genes, num_epochs=num_epochs, device=device,
        output_dir=output_dir, min_segment_confidence=min_segment_confidence,
        project_genes=project_genes,
    )
    
    # Determine genes to plot
    if plot_genes is None and project_genes:
        plot_genes = marker_genes[:10]
    
    save_results(adata_sp, adata_segment, output_dir, ad_ge=ad_ge, imgc=imgc, plot_genes=plot_genes)
    logger.info(f"=== Completed {library_id} ===")


def run_tangram_single(kwargs: dict) -> tuple[str, bool, str]:
    """Run tangram pipeline for a single sample (wrapper for batch processing)."""
    sample_id = kwargs.get("library_id", "unknown")
    try:
        run_tangram_pipeline(**kwargs)
        return (sample_id, True, "Success")
    except Exception as e:
        log_dir = pathlib.Path(kwargs.get("log_dir", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "crash.log", "w") as f:
            f.write(f"Crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {e}\n\n{traceback.format_exc()}")
        return (sample_id, False, str(e))


def run_tangram_batch(batch_params: list[dict], max_workers: int = 1) -> None:
    """Run tangram pipeline for multiple samples."""
    n_samples = len(batch_params)
    print(f"Starting batch: {n_samples} samples, {max_workers} workers")
    
    start_time = time.time()
    results = []
    
    if max_workers == 1:
        for i, params in enumerate(batch_params, 1):
            sample_id = params.get("library_id", f"sample_{i}")
            print(f"[{i}/{n_samples}] Processing {sample_id}...")
            results.append(run_tangram_single(params))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_tangram_single, p): p.get("library_id") for p in batch_params}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "✓" if result[1] else "✗"
                print(f"[{len(results)}/{n_samples}] {status} {result[0]}")
    
    successes = sum(1 for _, ok, _ in results if ok)
    print(f"\nCompleted in {time.time() - start_time:.1f}s: {successes}/{n_samples} succeeded")
    for sample_id, ok, msg in results:
        if not ok:
            print(f"  FAILED {sample_id}: {msg}")


# ============================================================================
# PLOTTING
# ============================================================================

def run_plot_single(
    mapped_ge_path: str,
    sp_h5ad_path: str,
    output_dir: str,
    library_id: str,
    genes: dict[str, list[str]],
    proportion_cutoff: float | dict[str, float] = 0.3,
    dpi: int = 300,
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Run gene expression visualization for a single sample."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adata_ge = load_mapped_ge(mapped_ge_path)
    adata_sp = load_adata_sp(sp_h5ad_path)
    
    if adata_ge.n_obs != adata_sp.n_obs:
        logger.warning(f"Shape mismatch: mapped_ge has {adata_ge.n_obs} spots, sp_tg has {adata_sp.n_obs} spots")
    
    for cell_type, gene_list in genes.items():
        if not gene_list:
            continue
        ct_cutoff = proportion_cutoff.get(cell_type, 0.3) if isinstance(proportion_cutoff, dict) else proportion_cutoff
        plot_genes_for_celltype(
            adata_ge=adata_ge, adata_sp=adata_sp, cell_type=cell_type, genes=gene_list,
            output_dir=output_dir, library_id=library_id, proportion_cutoff=ct_cutoff,
            dpi=dpi, figsize=figsize,
        )


def run_plot_batch(config: dict) -> None:
    """Run proportion plotting for multiple samples."""
    sp_h5ad_paths = config["sp_h5ad"]
    output_dirs = config["output_dir"]
    library_ids = config["library_id"]
    
    if not (len(sp_h5ad_paths) == len(output_dirs) == len(library_ids)):
        raise ValueError("sp_h5ad, output_dir, library_id must have same length")
    
    dpi = config.get("dpi", 300)
    figsize = tuple(config.get("figsize", (8, 8)))
    suffix = config.get("suffix", "")
    cmap = config.get("cmap", "viridis")
    perc = config.get("perc", 2.0)
    
    n_samples = len(sp_h5ad_paths)
    print(f"Processing {n_samples} samples")
    
    for i, (sp_path, out_dir, lib_id) in enumerate(zip(sp_h5ad_paths, output_dirs, library_ids), 1):
        print(f"[{i}/{n_samples}] Processing {pathlib.Path(sp_path).parent.name}...")
        try:
            plot_tangram_proportion(
                sp_h5ad_path=sp_path, output_dir=out_dir, library_id=lib_id,
                dpi=dpi, figsize=figsize, cmap=cmap, perc=perc, suffix=suffix,
            )
            print("  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()
