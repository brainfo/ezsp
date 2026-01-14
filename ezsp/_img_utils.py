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
from scipy import ndimage
from skimage import color, measure, morphology
from skimage.filters import frangi, threshold_multiotsu, threshold_otsu

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


def segment_villi(
    img: np.ndarray,
    min_area: int = 1000,
    max_area: int = 1000000,
    ridge_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    line_threshold: float | None = None,
    merge_kernel_size: int = 8,
    separation_dilation: int = 3,
    local_block_size: int = 501,
) -> np.ndarray:
    """
    Segment villi from histology image using combined background and border detection.
    
    Detects villi as tissue regions separated by:
    1. White/bright lumen background
    2. Purple border lines (detected using Frangi ridge filter)
    
    Args:
        img: RGB image array (H, W, 3) in uint8 or float format
        min_area: Minimum area in pixels for a valid villus
        max_area: Maximum area in pixels for a valid villus  
        ridge_sigmas: Sigma values for multi-scale Frangi ridge filter
        line_threshold: Threshold for ridge response (auto Otsu if None)
        merge_kernel_size: Kernel size for merging over-fragmented regions
        separation_dilation: Dilation radius for separation mask
        local_block_size: Block size for local adaptive thresholding (must be odd)
        
    Returns:
        Instance segmentation mask (H, W) where each villus has a unique label (0 = background)
    """
    logger.info("Starting villi segmentation using ridge + background detection")
    
    # Convert to float if needed
    if img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)
    
    # Step 1: Convert to LAB color space
    lab = color.rgb2lab(img_float)
    L, a = lab[:, :, 0], lab[:, :, 1]
    
    # Purple score for ridge detection: lower L and higher a = more purple
    purple_score = (-L / 100.0 + a / 128.0 + 1.0) / 2.0
    purple_score = np.clip(purple_score, 0, 1)
    
    logger.debug(f"Purple score range: [{purple_score.min():.3f}, {purple_score.max():.3f}]")
    
    # Step 2: Detect background using LOCAL adaptive thresholding
    # This handles uneven illumination across the tissue
    from skimage.filters import threshold_local
    
    # Ensure block_size is odd and not larger than image
    block_size = min(local_block_size, min(L.shape) - 1)
    if block_size % 2 == 0:
        block_size += 1
    
    local_thresh = threshold_local(L, block_size=block_size, method='gaussian', offset=-5)
    background_mask = L > local_thresh
    logger.info(f"Using local adaptive threshold with block_size={block_size}")
    
    # Step 3: Detect purple border LINES using Frangi ridge filter
    ridges = frangi(purple_score, sigmas=ridge_sigmas, black_ridges=False)
    
    if line_threshold is None:
        ridge_nonzero = ridges[ridges > 0]
        if len(ridge_nonzero) > 0:
            line_threshold = threshold_otsu(ridge_nonzero)
        else:
            line_threshold = 0.01
        logger.info(f"Auto-computed ridge threshold: {line_threshold:.6f}")
    
    border_lines = ridges > line_threshold
    logger.debug(f"Border line pixels: {border_lines.sum()}")
    
    # Step 4: Combine background and border lines as separation
    # Both white lumen AND purple borders separate individual villi
    separation = background_mask | border_lines
    separation_dilated = morphology.dilation(separation, morphology.disk(separation_dilation))
    
    # Step 5: Detect valid image area (exclude black borders)
    img_mean = np.mean(img, axis=2)
    valid_area = img_mean > 20
    
    # Villi = NOT separation AND in valid area
    villi_mask = (~separation_dilated) & valid_area
    
    # Step 6: Clean up villi mask
    # Remove small noise
    villi_cleaned = morphology.remove_small_objects(villi_mask.astype(bool), min_size=min_area // 4)
    
    # Fill small holes
    villi_filled = ndimage.binary_fill_holes(villi_cleaned)
    
    # Merge over-fragmented pieces using morphological closing
    villi_merged = morphology.closing(villi_filled, morphology.disk(merge_kernel_size))
    villi_merged = ndimage.binary_fill_holes(villi_merged)
    
    # Step 7: Label connected components
    labeled, num_features = ndimage.label(villi_merged)
    logger.info(f"Found {num_features} regions before filtering")
    
    # Step 8: Filter by size AND by mean brightness (reject white background regions)
    regions = measure.regionprops(labeled, intensity_image=L)
    valid_labels = []
    
    for region in regions:
        # Filter by size
        if not (min_area <= region.area <= max_area):
            logger.debug(f"Filtered region {region.label}: area={region.area} (outside range)")
            continue
        
        # Filter out white/background regions (mean L > 80 is too bright = background)
        mean_L = region.mean_intensity
        if mean_L > 80:
            logger.debug(f"Filtered region {region.label}: mean_L={mean_L:.1f} (too bright, likely background)")
            continue
            
        valid_labels.append(region.label)
    
    logger.info(f"After size and brightness filter: {len(valid_labels)} regions")
    
    # Step 9: Handle nested/contained regions - keep only outer boundaries
    # If region A is completely inside region B, remove A
    filtered_regions = [r for r in regions if r.label in valid_labels]
    
    # Sort by area (largest first) to process outer regions first
    filtered_regions.sort(key=lambda r: r.area, reverse=True)
    
    # Build a mask to track which pixels are already claimed by larger regions
    claimed_mask = np.zeros_like(labeled, dtype=bool)
    final_labels = []
    
    for region in filtered_regions:
        region_mask = labeled == region.label
        
        # Check if this region is mostly contained within already-claimed areas
        overlap = np.sum(region_mask & claimed_mask)
        containment_ratio = overlap / region.area
        
        if containment_ratio > 0.8:  # 80% contained = nested region, skip it
            logger.debug(f"Filtered region {region.label}: {containment_ratio:.1%} contained in larger region")
            continue
        
        final_labels.append(region.label)
        claimed_mask |= region_mask
    
    logger.info(f"After containment filter: {len(final_labels)} regions")
    
    # Create final mask with valid regions, relabeled sequentially
    final_mask = np.zeros_like(labeled)
    for new_label, old_label in enumerate(final_labels, start=1):
        final_mask[labeled == old_label] = new_label
    
    logger.info(f"Segmented {len(final_labels)} villi (area range: {min_area}-{max_area})")
    
    return final_mask