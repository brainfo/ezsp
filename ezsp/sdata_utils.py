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