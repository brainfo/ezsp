import numpy as np
import dask.array as da
from functools import partial

def normalize_block(block, p_low=1, p_high=99, gamma=0.5):
    result = np.empty(block.shape, dtype=np.uint8)
    for c in range(block.shape[0]):
        ch = block[c].astype(np.float32)
        p1, p99 = np.percentile(ch.ravel()[::10], [p_low, p_high])
        ch = np.clip(ch, p1, p99)
        result[c] = (((ch - p1) / (p99 - p1)) ** gamma * 255).astype(np.uint8)
    return result

def normalize_image(img, p_low=1, p_high=99, gamma=0.5, compute=False):
    """Normalize xarray or numpy image"""
    if isinstance(img, np.ndarray):
        return normalize_block(img, p_low, p_high, gamma)
    
    img_normalized = da.map_blocks(
        partial(normalize_block, p_low=p_low, p_high=p_high, gamma=gamma),
        img.data,
        dtype=np.uint8,
        meta=np.array((), dtype=np.uint8)
    )
    result = img.copy(data=img_normalized)
    return result.compute() if compute else result