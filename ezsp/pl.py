#!/usr/bin/env python

import logging
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData

from ._pp import load_tg_results as load_adata_sp

logger = logging.getLogger(__name__)


def normalize_ct_proportions(adata_sp: AnnData) -> list[str]:
    """
    Normalize cell type predictions and add to adata_sp.obs.
    
    Uses Tangram-style normalization: each cell type is normalized by dividing
    by its maximum value, so values range from 0 to 1 where 1 indicates the
    spot with highest enrichment for that cell type.
    
    Args:
        adata_sp: Spatial AnnData with tangram_ct_pred in obsm
        
    Returns:
        List of cell type names that were normalized and added to obs
    """
    if "tangram_ct_pred" not in adata_sp.obsm:
        raise ValueError("tangram_ct_pred not found in adata_sp.obsm")
    
    ct_pred = adata_sp.obsm["tangram_ct_pred"]
    ct_names = adata_sp.uns.get("tangram_ct_pred_names", [])
    
    if len(ct_names) != ct_pred.shape[1]:
        raise ValueError(
            f"Mismatch: {len(ct_names)} cell type names vs {ct_pred.shape[1]} columns"
        )
    
    # Normalize per cell type (column-wise max normalization)
    ct_proportion = ct_pred.copy().astype(np.float64)
    for i in range(ct_proportion.shape[1]):
        col_max = ct_proportion[:, i].max()
        if col_max > 0:
            ct_proportion[:, i] = ct_proportion[:, i] / col_max
    
    # Store normalized proportions
    adata_sp.obsm["tangram_ct_proportion"] = ct_proportion
    
    # Add individual columns to obs for plotting
    for i, ct in enumerate(ct_names):
        adata_sp.obs[ct] = ct_proportion[:, i] ## are these all inplace? and global?
    
    logger.info(f"Normalized {len(ct_names)} cell type proportions")
    return list(ct_names), adata_sp


def plot_tangram_proportion(
    sp_h5ad_path: str,
    output_dir: str,
    library_id: str,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    perc: float = 2.0,
    suffix: str = "",
) -> None:
    """
    Plot cell type proportions from Tangram predictions.
    
    Generates:
    1. Combined figure with all cell types
    2. Individual cell type proportion plots
    
    Args:
        sp_h5ad_path: Path to sp_tg.h5ad file
        output_dir: Directory to save output plots
        library_id: Library ID for spatial scatter
        dpi: Output image DPI
        figsize: Figure size for individual plots
        cmap: Colormap for proportion visualization
        perc: Percentile for vmin/vmax clipping
        suffix: Optional suffix for output filenames
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load spatial data
    logger.info(f"Loading {sp_h5ad_path}")
    adata_sp = load_adata_sp(sp_h5ad_path)
    
    # Normalize cell type proportions
    celltypes, adata_sp = normalize_ct_proportions(adata_sp)

    if not celltypes:
        logger.warning("No cell types found to plot")
        return
    
    logger.info(f"Plotting {len(celltypes)} cell types: {celltypes}")
    
    # Combined figure with all cell types
    n_cols = min(5, len(celltypes))
    n_rows = (len(celltypes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.09, 1.338))
    axes = np.atleast_1d(axes).flatten()
    
    for i, ct in enumerate(celltypes):
        values = adata_sp.obs[ct].values
        vmin = np.percentile(values, perc)
        vmax = np.percentile(values, 100 - perc)
        
        sq.pl.spatial_scatter(
            adata_sp,
            color=ct,
            shape=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=axes[i],
            library_id=library_id,
        )
        axes[i].set_title(f"{ct}")
    
    # Hide unused axes
    for j in range(len(celltypes), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / f"celltype_proportions{suffix}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"celltype_proportions{suffix}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined cell type proportion plot")
    
    # Individual cell type proportion plots
    ct_dir = output_dir / "celltype_proportions_individual"
    ct_dir.mkdir(exist_ok=True)
    
    for ct in celltypes:
        values = adata_sp.obs[ct].values
        vmin = np.percentile(values, perc)
        vmax = np.percentile(values, 100 - perc)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sq.pl.spatial_scatter(
            adata_sp,
            color=ct,
            shape=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            library_id=library_id,
            frameon=False,
        )
        ax.set_title(f"{ct} (normalized proportion)")
        
        fig.savefig(ct_dir / f"{ct}_proportion{suffix}.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(ct_dir / f"{ct}_proportion{suffix}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    
    logger.info(f"Saved {len(celltypes)} individual cell type proportion plots to {ct_dir}")

def load_mapped_ge(mapped_ge_path: str) -> AnnData:
    """Load mapped gene expression AnnData.
    
    Args:
        mapped_ge_path: Path to mapped_ge.h5ad
        
    Returns:
        AnnData with shape (n_spots, n_genes) containing imputed expression
    """
    adata = sc.read_h5ad(mapped_ge_path)
    logger.info(f"Loaded mapped_ge: {adata.shape} (spots x genes)")
    return adata

def normalize_gene_expression_for_plot(ad_ge, perc=2.0):
    """
    Normalize projected gene expression matching Tangram paper.
    
    1. Total count normalization
    2. Log1p
    3. Per-gene min-max scaling (with percentile clipping)
    """
    # Already done by tg.project_genes, but ensure it's there
    if "log_normalized" not in ad_ge.layers:
        sc.pp.normalize_total(ad_ge, target_sum=1e4)
        sc.pp.log1p(ad_ge)
    
    # For plotting each gene: min-max with percentile clipping
    X = ad_ge.X.toarray() if hasattr(ad_ge.X, 'toarray') else ad_ge.X
    
    for i in range(X.shape[1]):
        col = X[:, i]
        vmin = np.percentile(col, perc)
        vmax = np.percentile(col, 100 - perc)
        if vmax > vmin:
            X[:, i] = np.clip((col - vmin) / (vmax - vmin), 0, 1)
    
    return X

def get_celltype_mask(
    adata_sp: AnnData,
    cell_type: str,
    proportion_cutoff: float = 0.3,
) -> np.ndarray:
    """Get boolean mask for spots enriched in a cell type.
    
    Args:
        adata_sp: Spatial AnnData with tangram predictions
        cell_type: Cell type name (e.g., "STB")
        proportion_cutoff: Minimum proportion threshold
        
    Returns:
        Boolean mask array of shape (n_spots,)
    """
    # Check if cell type proportion is in obs columns (normalized proportions)
    proportions = normalize_ct_proportions(adata_sp)
    if cell_type in adata_sp.obs.columns:
        proportions = adata_sp.obs[cell_type].values
    elif "tangram_ct_pred" in adata_sp.obsm:
        # Fall back to obsm matrix
        ct_names = adata_sp.uns.get("tangram_ct_pred_names", [])
        if cell_type not in ct_names:
            raise ValueError(f"Cell type '{cell_type}' not found. Available: {ct_names}")
        ct_idx = ct_names.index(cell_type)
        # Use normalized proportions if available
        if "tangram_ct_pred_proportion" in adata_sp.obsm:
            proportions = adata_sp.obsm["tangram_ct_pred_proportion"][:, ct_idx]
        else:
            ct_pred = adata_sp.obsm["tangram_ct_pred"]
            ct_proportion = ct_pred.copy().astype(np.float64)
            for i in range(ct_proportion.shape[1]):
                col_max = ct_proportion[:, i].max()
                if col_max > 0:
                    ct_proportion[:, i] = ct_proportion[:, i] / col_max
            proportions = ct_proportion[:, ct_idx]
    else:
        raise ValueError(f"No cell type predictions found in adata_sp")
    
    mask = proportions >= proportion_cutoff
    logger.info(f"{cell_type}: {mask.sum()}/{len(mask)} spots with proportion >= {proportion_cutoff}")
    return mask


def plot_mapped_gene_expression(
    adata_ge: AnnData,
    adata_sp: AnnData,
    gene: str,
    cell_type: Optional[str] = None,
    proportion_cutoff: float = 0.3,
    output_dir: Optional[pathlib.Path] = None,
    library_id: Optional[str] = None,
    dpi: int = 300,
    figsize: tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    show_background: bool = True,
) -> tuple[float, float, np.ndarray]:
    """Plot mapped gene expression on spatial coordinates.
    
    Uses Tangram-style normalization: total count → log1p → percentile min-max.
    Color scale is fixed to [0, 1] since normalization scales to this range.
    
    Args:
        adata_ge: Mapped gene expression AnnData (from mapped_ge.h5ad)
        adata_sp: Spatial AnnData with cell type predictions (from sp_tg.h5ad)
        gene: Gene name to visualize
        cell_type: Optional cell type to filter by
        proportion_cutoff: Minimum proportion for cell type filtering
        output_dir: Directory to save plots (if None, don't save)
        library_id: Library ID for spatial metadata
        dpi: Output image DPI
        figsize: Figure size
        cmap: Colormap name
        show_background: Whether to show H&E background
        
    Returns:
        Tuple of (vmin, vmax, expression_values)
    """
    if gene not in adata_ge.var_names:
        logger.warning(f"Gene '{gene}' not found in mapped expression. Available: {adata_ge.n_vars} genes")
        return (0.0, 1.0, np.array([]))
    
    # Normalize expression using Tangram-style normalization
    X_normalized = normalize_gene_expression_for_plot(adata_ge.copy())
    
    # Get expression values for the specific gene
    gene_idx = list(adata_ge.var_names).index(gene)
    expr = X_normalized[:, gene_idx]
    
    # Check if this is a training gene
    is_training = False
    if "is_training" in adata_ge.var.columns:
        is_training = adata_ge.var.loc[gene, "is_training"]
    
    # Create working copy with expression
    adata_plot = adata_sp.copy()
    
    # Apply cell type filter if specified
    if cell_type is not None:
        mask = get_celltype_mask(adata_sp, cell_type, proportion_cutoff)
        # Set expression to NaN for filtered spots (they won't be plotted)
        expr_masked = expr.copy()
        expr_masked[~mask] = np.nan
        adata_plot.obs[f"{gene}_expr"] = expr_masked
        title_suffix = f" in {cell_type} (≥{proportion_cutoff:.0%})"
        filename_suffix = f"_{cell_type}"
    else:
        adata_plot.obs[f"{gene}_expr"] = expr
        title_suffix = ""
        filename_suffix = ""
    
    # Color scale is fixed to [0, 1] since normalization already scales to this range
    valid_expr = expr[~np.isnan(adata_plot.obs[f"{gene}_expr"])] if cell_type else expr
    if len(valid_expr) == 0:
        logger.warning(f"No valid expression values for {gene}")
        return (0.0, 1.0, np.array([]))
    
    computed_vmin = 0.0
    computed_vmax = 1.0
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    plot_kwargs = {
        "color": f"{gene}_expr",
        "cmap": cmap,
        "vmin": computed_vmin,
        "vmax": computed_vmax,
        "ax": ax,
        "frameon": False,
    }
    
    if show_background and "spatial" in adata_plot.uns:
        plot_kwargs["img_res_key"] = "lowres"
        plot_kwargs["img_alpha"] = 0.3
    
    if library_id:
        plot_kwargs["library_id"] = library_id
    
    sq.pl.spatial_scatter(adata_plot, **plot_kwargs)
    
    training_tag = " [training]" if is_training else " [imputed]"
    ax.set_title(f"{gene}{training_tag}{title_suffix}")
    
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"mapped_{gene}{filename_suffix}.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"mapped_{gene}{filename_suffix}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {gene}{filename_suffix} plot to {output_dir}")
    
    plt.close(fig)
    
    return (computed_vmin, computed_vmax, valid_expr)


def plot_genes_for_celltype(
    adata_ge: AnnData,
    adata_sp: AnnData,
    cell_type: str,
    genes: list[str],
    output_dir: pathlib.Path,
    library_id: Optional[str] = None,
    proportion_cutoff: float = 0.3,
    dpi: int = 300,
    figsize: tuple[int, int] = (8, 8),
) -> dict[str, tuple[float, float]]:
    """Plot multiple genes for a specific cell type.
    
    Uses Tangram-style normalization: total count → log1p → percentile min-max.
    Color scale is fixed to [0, 1] since normalization scales to this range.
    
    Args:
        adata_ge: Mapped gene expression AnnData
        adata_sp: Spatial AnnData with cell type predictions
        cell_type: Cell type to filter by
        genes: List of genes to plot
        output_dir: Output directory
        library_id: Library ID
        proportion_cutoff: Minimum proportion threshold
        dpi: Output DPI
        figsize: Figure size
        
    Returns:
        Dict of gene -> (vmin, vmax) with fixed (0, 1) ranges
    """
    gene_ranges = {}
    
    # Create cell-type specific output directory
    ct_dir = output_dir / f"mapped_expression_{cell_type}"
    ct_dir.mkdir(parents=True, exist_ok=True)
    
    for gene in genes:
        if gene not in adata_ge.var_names:
            logger.warning(f"Gene '{gene}' not found in mapped expression, skipping")
            continue
        
        computed_vmin, computed_vmax, _ = plot_mapped_gene_expression(
            adata_ge=adata_ge,
            adata_sp=adata_sp,
            gene=gene,
            cell_type=cell_type,
            proportion_cutoff=proportion_cutoff,
            output_dir=ct_dir,
            library_id=library_id,
            dpi=dpi,
            figsize=figsize,
        )
        gene_ranges[gene] = (computed_vmin, computed_vmax)
    
    # Also create a combined figure
    available_genes = [g for g in genes if g in adata_ge.var_names]
    if len(available_genes) > 1:
        n_cols = min(4, len(available_genes))
        n_rows = (len(available_genes) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        mask = get_celltype_mask(adata_sp, cell_type, proportion_cutoff)
        adata_plot = adata_sp.copy()
        
        # Normalize once for all genes in the combined plot
        X_normalized = normalize_gene_expression_for_plot(adata_ge.copy())
        
        for i, gene in enumerate(available_genes):
            gene_idx = list(adata_ge.var_names).index(gene)
            expr = X_normalized[:, gene_idx]
            
            expr_masked = expr.copy()
            expr_masked[~mask] = np.nan
            adata_plot.obs[f"{gene}_expr"] = expr_masked
            
            sq.pl.spatial_scatter(
                adata_plot, 
                color=f"{gene}_expr", 
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                ax=axes[i],
                frameon=False,
            )
            axes[i].set_title(gene)
        
        for j in range(len(available_genes), len(axes)):
            axes[j].axis("off")
        
        plt.suptitle(f"Mapped expression in {cell_type} (proportion ≥ {proportion_cutoff:.0%})")
        plt.tight_layout()
        fig.savefig(ct_dir / f"combined_{cell_type}.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(ct_dir / f"combined_{cell_type}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved combined plot for {cell_type}")
    
    return gene_ranges

def run_single(
    mapped_ge_path: str,
    sp_h5ad_path: str,
    output_dir: str,
    library_id: str,
    genes: dict[str, list[str]],
    proportion_cutoff: float | dict[str, float] = 0.3,
    dpi: int = 300,
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Run visualization for a single sample.
    
    Uses Tangram-style normalization: total count → log1p → percentile min-max.
    Color scale is fixed to [0, 1] since normalization scales to this range.
    
    Args:
        mapped_ge_path: Path to mapped_ge.h5ad
        sp_h5ad_path: Path to sp_tg.h5ad (for cell type proportions)
        output_dir: Output directory
        library_id: Library ID
        genes: Dict of {cell_type: [gene_list]}
        proportion_cutoff: Single float or dict of {cell_type: cutoff}
        dpi: Output DPI
        figsize: Figure size
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata_ge = load_mapped_ge(mapped_ge_path)
    adata_sp = load_adata_sp(sp_h5ad_path)
    
    # Check alignment
    if adata_ge.n_obs != adata_sp.n_obs:
        logger.warning(f"Shape mismatch: mapped_ge has {adata_ge.n_obs} spots, sp_tg has {adata_sp.n_obs} spots")
    
    for cell_type, gene_list in genes.items():
        if not gene_list:
            continue
        
        # Get cell-type specific cutoff or use default
        if isinstance(proportion_cutoff, dict):
            ct_cutoff = proportion_cutoff.get(cell_type, 0.3)
        else:
            ct_cutoff = proportion_cutoff
        
        plot_genes_for_celltype(
            adata_ge=adata_ge,
            adata_sp=adata_sp,
            cell_type=cell_type,
            genes=gene_list,
            output_dir=output_dir,
            library_id=library_id,
            proportion_cutoff=ct_cutoff,
            dpi=dpi,
            figsize=figsize,
        )

def run_batch(config: dict) -> None:
    """
    Run proportion plotting for multiple samples.
    
    Args:
        config: Configuration dictionary with sp_h5ad, output_dir, library_id lists
    """
    sp_h5ad_paths = config["sp_h5ad"]
    output_dirs = config["output_dir"]
    library_ids = config["library_id"]
    
    if len(sp_h5ad_paths) != len(output_dirs) or len(sp_h5ad_paths) != len(library_ids):
        raise ValueError(
            f"sp_h5ad ({len(sp_h5ad_paths)}), output_dir ({len(output_dirs)}), and "
            f"library_id ({len(library_ids)}) must have the same length"
        )
    
    dpi = config.get("dpi", 300)
    figsize = tuple(config.get("figsize", (1.9475, 1.6725)))
    suffix = config.get("suffix", "")
    cmap = config.get("cmap", "viridis")
    perc = config.get("perc", 2.0)
    
    n_samples = len(sp_h5ad_paths)
    print(f"Processing {n_samples} samples")
    
    for i, (sp_path, out_dir, lib_id) in enumerate(
        zip(sp_h5ad_paths, output_dirs, library_ids), 1
    ):
        print(f"[{i}/{n_samples}] Processing {pathlib.Path(sp_path).parent.name}...")
        try:
            plot_tangram_proportion(
                sp_h5ad_path=sp_path,
                output_dir=out_dir,
                library_id=lib_id,
                dpi=dpi,
                figsize=figsize,
                cmap=cmap,
                perc=perc,
                suffix=suffix,
            )
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()