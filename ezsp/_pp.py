import logging
import pathlib
import time

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import yaml
from anndata import AnnData

from ._img_utils import normalize_image

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str | None = None, verbose: bool = False) -> pathlib.Path | None:
    """Configure logging. If log_dir provided, logs to file; otherwise console only."""
    logging.root.handlers = []
    
    handlers = []
    log_file = None
    
    if log_dir:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"tangram_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        ))
        handlers.append(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    handlers.append(console_handler)
    
    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    
    if log_file:
        print(f"Logging to: {log_file}")
    return log_file

def prepare_single_cell_reference(
    sc_path: str,
    n_top_genes: int = 100,
    groupby: str = "major_celltype",
    min_cells_per_type: int = 10,
    cell_types: set[str] | None = None,
    cell_types_proportion: dict[str, float] | None = None,
) -> tuple[AnnData, list[str]]:
    """Load and prepare single-cell reference data for Tangram."""
    adata_sc = sc.read_h5ad(sc_path)
    logger.info(f"Loaded single-cell reference: {adata_sc.shape}")
    
    celltype_counts = adata_sc.obs[groupby].value_counts()
    all_valid_celltypes = set(celltype_counts[celltype_counts >= min_cells_per_type].index.tolist())
    if cell_types is not None:
        valid_celltypes = all_valid_celltypes & cell_types
    else:
        valid_celltypes = all_valid_celltypes
    removed = celltype_counts[celltype_counts < min_cells_per_type]
    
    if len(removed) > 0:
        logger.warning(f"Filtering {len(removed)} cell types with < {min_cells_per_type} cells: {dict(removed)}")
    
    adata_sc = adata_sc[adata_sc.obs[groupby].isin(valid_celltypes)].copy()
    logger.info(f"After filtering: {adata_sc.shape}, {len(valid_celltypes)} cell types")
    
    if cell_types_proportion is not None:
        ct_counts = {}
        for ct in valid_celltypes:
            ct_mask = adata_sc.obs[groupby] == ct
            ct_counts[ct] = ct_mask.sum()
        
        limiting_ratios = {
            ct: ct_counts[ct] / prop 
            for ct, prop in cell_types_proportion.items() 
            if prop > 0 and ct_counts[ct] > 0
        }
        
        if not limiting_ratios:
            logger.warning("No valid cell types for proportional sampling")
        else:
            limiting_ct = min(limiting_ratios, key=limiting_ratios.get)
            total_output = int(limiting_ratios[limiting_ct])
            logger.info(f"Limiting cell type: {limiting_ct} (ratio={limiting_ratios[limiting_ct]:.1f})")
            logger.info(f"Total output size: {total_output}")
            
            sampled_indices = []
            for ct, proportion in cell_types_proportion.items():
                ct_mask = adata_sc.obs[groupby] == ct
                ct_indices = adata_sc.obs_names[ct_mask].tolist()
                n_target = int(total_output * proportion)
                
                if n_target > 0 and len(ct_indices) > 0:
                    sampled = np.random.choice(ct_indices, size=n_target, replace=False)
                    sampled_indices.extend(sampled)
                    logger.info(f"Sampled {n_target}/{len(ct_indices)} cells for {ct} (target proportion={proportion})")
            
            adata_sc = adata_sc[sampled_indices].copy()
            
            actual_props = adata_sc.obs[groupby].value_counts(normalize=True)
            logger.info(f"After proportional sampling: {adata_sc.shape}")
            logger.info(f"Achieved proportions: {dict(actual_props)}")
    
    if "log_norm" not in adata_sc.layers:
        adata_sc.layers["log_norm"] = np.log1p(sc.pp.normalize_total(adata_sc, target_sum=1e4, inplace=False)['X'])

    sc.tl.rank_genes_groups(adata_sc, groupby=groupby, use_raw=False, layer="log_norm")
    
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:n_top_genes, :]
    marker_genes = np.unique(markers_df.melt().value.values).tolist()
    logger.info(f"Identified {len(marker_genes)} marker genes")
    
    if "raw" not in adata_sc.layers and adata_sc.X.dtype == np.int16:
        adata_sc.layers["raw"] = adata_sc.X.copy()
    adata_sc.X = adata_sc.layers["raw"].copy()
    return adata_sc, marker_genes

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
    img_hires = normalize_image(img_hires, compute=True)
    img_hires = np.moveaxis(img_hires.values, 0, -1)
    
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

def load_tg_results(h5ad_path: pathlib.Path) -> AnnData:
    """Load adata_sp with all tangram data restored from TSV backups."""
    import ast
    
    h5ad_path = pathlib.Path(h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    
    base_path = h5ad_path.with_suffix("")
    
    ct_pred_tsv = pathlib.Path(str(base_path) + ".tangram_ct_pred.tsv")
    if ct_pred_tsv.exists():
        df = pd.read_csv(ct_pred_tsv, sep="\t", index_col=0)
        adata.obsm["tangram_ct_pred"] = df.values
        adata.uns["tangram_ct_pred_names"] = list(df.columns)
        logger.debug(f"Restored tangram_ct_pred from {ct_pred_tsv.name}")
    
    ct_count_tsv = pathlib.Path(str(base_path) + ".tangram_ct_count.tsv")
    if ct_count_tsv.exists():
        df = pd.read_csv(ct_count_tsv, sep="\t", index_col=0)
        if "centroids" in df.columns:
            df["centroids"] = df["centroids"].apply(
                lambda x: np.array(ast.literal_eval(x), dtype=object) if isinstance(x, str) and x.startswith("[") else np.array([], dtype=object)
            )
        adata.uns["tangram_ct_count"] = df
        logger.debug(f"Restored tangram_ct_count from {ct_count_tsv.name}")
    
    cell_seg_tsv = pathlib.Path(str(base_path) + ".tangram_cell_segmentation.tsv")
    if cell_seg_tsv.exists():
        df = pd.read_csv(cell_seg_tsv, sep="\t")
        adata.uns["tangram_cell_segmentation"] = df
        logger.debug(f"Restored tangram_cell_segmentation from {cell_seg_tsv.name}")
    
    img_features_tsv = pathlib.Path(str(base_path) + ".image_features.tsv")
    if img_features_tsv.exists():
        df = pd.read_csv(img_features_tsv, sep="\t", index_col=0)
        df.index = adata.obs_names
        adata.obsm["image_features"] = df
        logger.debug(f"Restored image_features from {img_features_tsv.name}")
    
    return adata