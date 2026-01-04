from pathlib import Path
import shutil

import spatialdata as sd

def safe_update_sdata(sdata, new_path, old_path=None):
    """Write to new location, verify, then optionally replace old."""
    new_path = Path(new_path)
    
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
    
    # Handle DataFrames with non-string object columns (e.g., image_features with tuples)
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
    
    # Handle Tangram results
    if "tangram_ct_pred" in adata.obsm:
        ct_names = adata.uns.get("tangram_ct_pred_names", [f"ct_{i}" for i in range(adata.obsm["tangram_ct_pred"].shape[1])])
        ct_df = pd.DataFrame(adata.obsm["tangram_ct_pred"], index=adata.obs_names, columns=ct_names)
        ct_df.to_csv(path.with_suffix(".tangram_ct_pred.tsv"), sep="\t")
        removed_obsm["tangram_ct_pred"] = adata.obsm["tangram_ct_pred"]
        del adata.obsm["tangram_ct_pred"]
    
    if "tangram_ct_pred_names" in adata.uns:
        removed_uns["tangram_ct_pred_names"] = adata.uns["tangram_ct_pred_names"]
        del adata.uns["tangram_ct_pred_names"]
    
    adata.write_h5ad(path)
    
    # Restore for runtime use
    for key, val in removed_obsm.items():
        adata.obsm[key] = val
    for key, val in removed_uns.items():
        adata.uns[key] = val