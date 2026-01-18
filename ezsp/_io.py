import pathlib
import shutil
from pathlib import Path

import pandas as pd
import spatialdata as sd
from anndata import AnnData

def rechunk_datatree(dt, chunk_size=4096):
    """Rechunk all scales in DataTree for zarr compatibility."""
    import xarray as xr
    
    new_nodes = {}
    for path, node in dt.subtree_with_keys:
        if node.has_data and "image" in node.ds.data_vars:
            ds = node.ds
            img = ds["image"]
            
            # Regular chunks: 1 per channel, chunk_size or image size for y/x
            chunks = {
                "c": 1,
                "y": min(chunk_size, img.sizes["y"]),
                "x": min(chunk_size, img.sizes["x"])
            }
            img_rechunked = img.chunk(chunks)
            new_nodes[path] = ds.assign(image=img_rechunked)
        else:
            new_nodes[path] = xr.Dataset()
    
    return xr.DataTree.from_dict(new_nodes)

def safe_update_sdata(sdata, new_path, rechunk_img=False, old_path=None):
    """Write to new location, verify, then optionally replace old."""
    new_path = Path(new_path)

    if rechunk_img:
        ## rechunk all image elements
        for key in sdata.images.keys():
            sdata.images[key] = rechunk_datatree(sdata.images[key], chunk_size=4096)
    
    # 1. Write to new location
    sdata.write(new_path)
    
    # 2. Verify by reading back
    sdata_verify = sd.read_zarr(new_path)
    
    # 3. Check element counts match
    assert set(sdata.images.keys()) == set(sdata_verify.images.keys())
    assert set(sdata.labels.keys()) == set(sdata_verify.labels.keys())
    assert set(sdata.shapes.keys()) == set(sdata_verify.shapes.keys())
    assert set(sdata.tables.keys()) == set(sdata_verify.tables.keys())
    
    # 4. Only delete old after verification passes
    if old_path and Path(old_path).exists():
        shutil.rmtree(old_path)
    
    return sdata_verify

def save_adata_safe(adata: AnnData, path: pathlib.Path) -> None:
    """Save adata to h5ad, exporting non-serializable obsm/uns to TSV."""
    path = pathlib.Path(path)
    
    removed_obsm = {}
    removed_uns = {}
    
    # Handle DataFrames with non-string object columns
    for key in list(adata.obsm.keys()):
        obj = adata.obsm[key]
        if not isinstance(obj, pd.DataFrame):
            continue
        
        problematic = False
        for col in obj.columns:
            if obj[col].dtype == object:
                sample = obj[col].dropna()
                if len(sample) > 0 and not isinstance(sample.iloc[0], str):
                    problematic = True
                    break
        
        if problematic:
            df = obj.copy()
            for col in df.columns:
                if df[col].dtype == object:
                    first_valid = df[col].dropna()
                    if len(first_valid) > 0 and isinstance(first_valid.iloc[0], (tuple, list)):
                        expanded = df[col].apply(pd.Series)
                        expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
                        df = df.drop(columns=[col]).join(expanded)
            df.to_csv(path.with_suffix(f".{key}.tsv"), sep="\t")
            removed_obsm[key] = adata.obsm[key]
            del adata.obsm[key]
    
    # Handle Tangram ct_pred results
    if "tangram_ct_pred" in adata.obsm:
        ct_names = adata.uns.get("tangram_ct_pred_names", [f"ct_{i}" for i in range(adata.obsm["tangram_ct_pred"].shape[1])])
        ct_df = pd.DataFrame(adata.obsm["tangram_ct_pred"], index=adata.obs_names, columns=ct_names)
        ct_df.to_csv(path.with_suffix(".tangram_ct_pred.tsv"), sep="\t")
        removed_obsm["tangram_ct_pred"] = adata.obsm["tangram_ct_pred"]
        del adata.obsm["tangram_ct_pred"]
    
    if "tangram_ct_pred_names" in adata.uns:
        removed_uns["tangram_ct_pred_names"] = adata.uns["tangram_ct_pred_names"]
        del adata.uns["tangram_ct_pred_names"]
    
    # Handle tangram_ct_count
    if "tangram_ct_count" in adata.uns and isinstance(adata.uns["tangram_ct_count"], pd.DataFrame):
        df = adata.uns["tangram_ct_count"].copy()
        if "centroids" in df.columns:
            df["centroids"] = df["centroids"].apply(lambda x: str(x.tolist()) if hasattr(x, 'tolist') else str(x))
        df.to_csv(path.with_suffix(".tangram_ct_count.tsv"), sep="\t")
        removed_uns["tangram_ct_count"] = adata.uns["tangram_ct_count"]
        del adata.uns["tangram_ct_count"]
    
    if "tangram_cell_segmentation" in adata.uns and isinstance(adata.uns["tangram_cell_segmentation"], pd.DataFrame):
        adata.uns["tangram_cell_segmentation"].to_csv(
            path.with_suffix(".tangram_cell_segmentation.tsv"), sep="\t", index=False
        )
        removed_uns["tangram_cell_segmentation"] = adata.uns["tangram_cell_segmentation"]
        del adata.uns["tangram_cell_segmentation"]
    
    adata.write_h5ad(path)
    
    for key, val in removed_obsm.items():
        adata.obsm[key] = val
    for key, val in removed_uns.items():
        adata.uns[key] = val