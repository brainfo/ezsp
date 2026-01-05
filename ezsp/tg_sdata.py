import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel, Image2DModel
import squidpy as sq
import numpy as np

from ._img_utils import normalize_image

def segment4sdata(
    sdata: sd.SpatialData,
    scale: int = 0,
    save: str = "",
    image_key: str = "stain",
    points_key: str = "bin1_genes",  # Stereo-seq bin1 points
) -> sd.SpatialData:
    """Watershed segmentation and aggregate Stereo-seq bin1 to cells."""
    import os
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    # 1. Segment from HE/ssDNA image
    img_da = sdata.images[image_key][f"scale{scale}"]["image"]
    
    # Get lazy normalized image for storage, and computed array for processing
    img_norm_lazy = normalize_image(img_da, compute=False)  # Lazy dask-backed DataArray
    img_norm_computed = img_norm_lazy.compute().values      # Numpy array for segmentation
    
    # Store normalized image as proper SpatialData image (dask-backed DataTree)
    original_transforms = sd.transformations.get_transformation(img_da, get_all=True)
    sdata.images[f"{image_key}_norm"] = Image2DModel.parse(
        img_norm_lazy.data,  # dask array
        dims=img_da.dims,
        transformations=original_transforms,
    )
    
    from .segment import compute_segmentation_thresholds, segment
    imgc = sq.im.ImageContainer(img=img_norm_computed)
    thresholds = compute_segmentation_thresholds(imgc["image"], n_classes=3, output_dir=None)
    imgc = segment(img=img_norm_computed, thresh=thresholds[1])

    # 2. Extract mask and add as labels
    seg_mask = imgc["segmented_watershed"].values.squeeze()
    from spatialdata.models import Labels2DModel
    sdata.labels["cell_labels"] = Labels2DModel.parse(
        seg_mask, 
        transformations=original_transforms,
    )
    
    # 3. Convert labels to polygons
    cell_shapes = sd.to_polygons(sdata.labels["cell_labels"])
    sdata.shapes["cells"] = cell_shapes
    
    # 4. Assign points to cells using labels
    # Stereo-seq points don't have gene column - use labels to assign points to cells
    points_df = sdata.points[points_key].compute()
    
    # Get cell label for each point by looking up in the segmentation mask
    # Coordinates need to be integers for indexing
    y_coords = points_df["y"].values.astype(int)
    x_coords = points_df["x"].values.astype(int)
    
    # Clip to image bounds
    y_coords = np.clip(y_coords, 0, seg_mask.shape[0] - 1)
    x_coords = np.clip(x_coords, 0, seg_mask.shape[1] - 1)
    
    # Assign cell labels to points
    cell_ids = seg_mask[y_coords, x_coords]
    points_df["cell_id"] = cell_ids
    
    # Filter out points not in any cell (label 0 = background)
    points_in_cells = points_df[points_df["cell_id"] > 0]
    
    # Count points per cell
    cell_counts = points_in_cells.groupby("cell_id").size().reset_index(name="n_transcripts")
    
    # Store assignment info in sdata for downstream use
    sdata.attrs["points_to_cells"] = {
        "points_key": points_key,
        "n_points_total": len(points_df),
        "n_points_assigned": len(points_in_cells),
        "n_cells_with_points": len(cell_counts),
    }
    
    if save:
        from . import _io
        _io.safe_update_sdata(sdata, save)
    return sdata

