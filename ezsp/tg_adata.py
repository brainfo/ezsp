import logging
import os
import pathlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import squidpy as sq
import tangram as tg
from anndata import AnnData

from ._img_utils import compute_segmentation_thresholds, normalize_image
from ._io import save_adata_safe
from ._pp import load_config, prepare_for_squidpy, prepare_single_cell_reference, setup_logging
from .pl import normalize_ct_proportions, plot_mapped_gene_expression

logger = logging.getLogger(__name__)




def segment_and_extract_features(
    adata: AnnData | pathlib.Path,
    library_id: str,
    thresh: float,
    n_jobs: int = -1,
) -> tuple[AnnData, sq.im.ImageContainer]:
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
    if isinstance(adata, pathlib.Path):
        adata = sc.read_h5ad(adata)
    else:
        sq.im.calculate_image_features(
            adata, imgc, layer="image", key_added="image_features",
            features_kwargs=features_kwargs, features="segmentation",
            mask_circle=True, n_jobs=n_jobs,
        )
    
        adata.obs["cell_count"] = adata.obsm["image_features"]["segmentation_label"]
        logger.info(f"Extracted cell counts. Total: {adata.obs['cell_count'].sum()}")
    return adata, imgc

def normalize_ct_pred_tangram_style(ct_pred: np.ndarray) -> np.ndarray:
    """
    Normalize cell type predictions following Tangram's methodology.
    
    Tangram normalizes per cell type by dividing by the max value,
    NOT per spot to sum to 1. This ensures each cell type's values
    range from 0 to 1, where 1 indicates the spot with highest
    enrichment for that cell type.
    
    Args:
        ct_pred: Array of shape (n_spots, n_celltypes)
        
    Returns:
        Normalized array where each column max = 1.0
    """
    ct_pred_norm = ct_pred.copy().astype(np.float64)
    for i in range(ct_pred_norm.shape[1]):
        col_max = ct_pred_norm[:, i].max()
        if col_max > 0:
            ct_pred_norm[:, i] = ct_pred_norm[:, i] / col_max
    return ct_pred_norm


def normalize_ct_pred_proportion_style(ct_pred: np.ndarray) -> np.ndarray:
    """
    Alternative normalization: per-spot proportions summing to 1.
    
    This gives cell type composition per spot, useful when you want
    to know "what fraction of this spot is cell type X".
    
    Args:
        ct_pred: Array of shape (n_spots, n_celltypes)
        
    Returns:
        Normalized array where each row sums to 1.0
    """
    ct_pred_norm = ct_pred.copy().astype(np.float64)
    row_sums = ct_pred_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    ct_pred_norm = ct_pred_norm / row_sums
    return ct_pred_norm

def run_tangram_deconvolution(
    adata_sc: AnnData,
    adata_sp: AnnData,
    genes: list[str],
    annotation: str = "major_celltype",
    num_epochs: int = 1000,
    device: str = "cpu",
    random_state: int = 404,
    output_dir: Optional[pathlib.Path] = None,
    min_segment_confidence: float = 0.0,
    project_genes: bool = True,
    test_genes: list[str] | None = None,
) -> tuple[AnnData, AnnData, AnnData | None]:
    """
    Run Tangram spatial deconvolution with proper normalization.
    
    Args:
        adata_sc: Single-cell reference AnnData
        adata_sp: Spatial AnnData
        genes: Training genes for mapping
        annotation: Cell type annotation column in adata_sc.obs
        num_epochs: Training epochs
        device: 'cpu' or 'cuda:0'
        random_state: Random seed
        output_dir: Output directory for results
        min_segment_confidence: Minimum probability threshold for segment assignment
        project_genes: Whether to project gene expression onto space
        test_genes: Optional list of test genes for validation
        
    Returns:
        tuple: (adata_sp with ct predictions, adata_segment, ad_ge or None)
    """
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
    
    # === CELL TYPE PREDICTIONS ===
    # project_cell_annotations computes: M.T @ one_hot(annotation)
    # Then normalizes per cell type by max value
    tg.project_cell_annotations(ad_map, adata_sp, annotation=annotation)
    
    ct_names = adata_sp.uns["tangram_ct_pred_names"]
    ct_pred_raw = adata_sp.obsm["tangram_ct_pred"].copy()
    
    # Store both raw and normalized versions
    # Raw: actual mapped cell counts (can exceed 1 in constrained mode)
    # Tangram-norm: per-celltype max normalization (0-1 per celltype)
    # Proportion-norm: per-spot proportions (sums to 1 per spot)
    
    ct_pred_tangram = normalize_ct_pred_tangram_style(ct_pred_raw)
    ct_pred_proportion = normalize_ct_pred_proportion_style(ct_pred_raw)
    
    # Store in obsm
    adata_sp.obsm["tangram_ct_pred_raw"] = ct_pred_raw
    adata_sp.obsm["tangram_ct_pred"] = ct_pred_tangram  # Default: Tangram style
    adata_sp.obsm["tangram_ct_pred_proportion"] = ct_pred_proportion
    
    # Add to obs for easy plotting (using Tangram-style normalization)
    for i, ct in enumerate(ct_names):
        adata_sp.obs[f"{ct}_raw"] = ct_pred_raw[:, i]
        adata_sp.obs[f"{ct}"] = ct_pred_tangram[:, i]  # 0-1 normalized
        adata_sp.obs[f"{ct}_proportion"] = ct_pred_proportion[:, i]
    
    # Log normalization statistics
    logger.info("Cell type prediction statistics (Tangram-style normalized):")
    for i, ct in enumerate(ct_names):
        raw_max = ct_pred_raw[:, i].max()
        norm_max = ct_pred_tangram[:, i].max()
        prop_max = ct_pred_proportion[:, i].max()
        logger.info(f"  {ct}: raw_max={raw_max:.3f}, tangram_norm_max={norm_max:.3f}, prop_max={prop_max:.3f}")
    
    # === GENE EXPRESSION PROJECTION ===
    ad_ge = None
    if project_genes:
        logger.info("Projecting gene expression onto space...")
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
        
        # Normalize projected gene expression (normalized mRNA counts as in paper)
        # The paper uses log1p normalization for visualization
        if ad_ge.X is not None:
            # Store raw projected counts
            ad_ge.layers["projected_raw"] = ad_ge.X.copy()
            
            # Normalize per spot (total count normalization)
            sc.pp.normalize_total(ad_ge, target_sum=1e4)
            ad_ge.layers["normalized"] = ad_ge.X.copy()
            
            # Log transform for visualization
            sc.pp.log1p(ad_ge)
            ad_ge.layers["log_normalized"] = ad_ge.X.copy()
            
            logger.info(f"Projected {ad_ge.n_vars} genes onto {ad_ge.n_obs} spots")
        
        # Validate with test genes if provided
        if test_genes is not None:
            test_genes_available = [g for g in test_genes if g in ad_ge.var_names and g in adata_sp.var_names]
            if test_genes_available:
                df_test = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)
                adata_sp.uns["tangram_gene_scores"] = df_test
                logger.info(f"Computed gene expression scores for {len(test_genes_available)} test genes")
    
    # === SEGMENT-LEVEL DECONVOLUTION ===
    tg.create_segment_cell_df(adata_sp)
    tg.count_cell_annotations(ad_map, adata_sc, adata_sp, annotation=annotation)
    adata_segment = tg.deconvolve_cell_annotations(adata_sp)
    
    # Filter low-confidence segments
    if min_segment_confidence > 0:
        probs = adata_segment.X
        max_probs = probs.max(axis=1)
        low_conf_mask = max_probs < min_segment_confidence
        n_unassigned = low_conf_mask.sum()
        adata_segment.obs["cluster"] = np.where(
            low_conf_mask, "Unassigned", adata_segment.obs["cluster"]
        )
        logger.info(f"Segment confidence filter: {n_unassigned}/{adata_segment.n_obs} marked as 'Unassigned'")
    
    logger.info("Completed segment-level deconvolution")
    
    return adata_sp, adata_segment, ad_ge

def save_results(
    adata_sp: AnnData,
    adata_segment: AnnData,
    output_dir: pathlib.Path,
    ad_ge: AnnData | None = None,
    imgc: Optional[sq.im.ImageContainer] = None,
    celltypes: Optional[list[str]] = None,
    plot_genes: list[str] | None = None,
) -> None:
    """Save deconvolution results and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if celltypes is None:
        celltypes = ["STB", "CTB", "EVT", "Fibroblast", "Myofibroblast", "VEC", "Hofbauer cells"]
    
    # Plot cell type proportions
    ct_names, _ = normalize_ct_proportions(adata_sp)
    for ct in celltypes:
        if ct in ct_names:
            values = adata_sp.obs[ct].values
            vmin, vmax = np.percentile(values, [2.0, 98.0])
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sq.pl.spatial_scatter(adata_sp, color=ct, cmap="viridis", vmin=vmin, vmax=vmax, ax=ax, frameon=False)
            fig.savefig(output_dir / f"{ct}_proportion.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(output_dir / f"{ct}_proportion.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    logger.info("Saved cell type proportion plots")
    
    # Plot gene expression if available
    if ad_ge is not None and plot_genes is not None:
        for gene in plot_genes:
            plot_mapped_gene_expression(ad_ge, adata_sp, gene, output_dir=output_dir)
    
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
    
    # Save h5ad files
    save_adata_safe(adata_sp, output_dir / "sp_tg.h5ad")
    save_adata_safe(adata_segment, output_dir / "segment.h5ad")
    
    if ad_ge is not None:
        ad_ge.write_h5ad(output_dir / "projected_genes.h5ad")
        logger.info("Saved projected gene expression")
    
    # Save summary statistics
    ct_names = adata_sp.uns.get("tangram_ct_pred_names", [])
    if ct_names:
        stats_data = []
        for ct in ct_names:
            if ct in adata_sp.obs.columns:
                stats_data.append({
                    "cell_type": ct,
                    "mean_tangram": adata_sp.obs[ct].mean(),
                    "max_tangram": adata_sp.obs[ct].max(),
                    "mean_proportion": adata_sp.obs[f"{ct}_proportion"].mean() if f"{ct}_proportion" in adata_sp.obs else np.nan,
                    "max_proportion": adata_sp.obs[f"{ct}_proportion"].max() if f"{ct}_proportion" in adata_sp.obs else np.nan,
                    "mean_raw": adata_sp.obs[f"{ct}_raw"].mean() if f"{ct}_raw" in adata_sp.obs else np.nan,
                    "max_raw": adata_sp.obs[f"{ct}_raw"].max() if f"{ct}_raw" in adata_sp.obs else np.nan,
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_dir / "celltype_statistics.tsv", sep="\t", index=False)
        logger.info("Saved cell type statistics")
    
    logger.info(f"Saved results to {output_dir}")
