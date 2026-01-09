import logging
import os
import pathlib
from functools import partial
from typing import Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
import xarray
from skimage.filters import threshold_multiotsu

logger = logging.getLogger(__name__)

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

def compute_segmentation_thresholds(
    img: np.ndarray,
    n_classes: int = 3,
    output_dir: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """Compute multi-Otsu thresholds for image segmentation."""
    thresholds = threshold_multiotsu(np.array(img), classes=n_classes)
    logger.info(f"Computed multi-Otsu thresholds: {thresholds}")
    
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "histogram.pdf").exists():
            logger.info("Histogram already exists, skipping")
            return thresholds
        
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(img).ravel(), bins=50, range=(0, 256), alpha=0.7)
        plt.vlines(thresholds, 0, plt.gca().get_ylim()[1], color="r", linewidth=2)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Image Histogram with Multi-Otsu Thresholds")
        plt.savefig(output_dir / "histogram.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "histogram.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved histogram to {output_dir}")
    
    return thresholds

def segment(img: np.ndarray | xarray.DataArray,
    thresh: float,
) -> sq.im.ImageContainer:
    """Perform watershed segmentation"""
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    imgc = sq.im.ImageContainer(img=img)
    sq.im.segment(img=imgc, layer="image", method="watershed", thresh=thresh, geq=False)

    return imgc