import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel
import squidpy as sq
import numpy as np

from .img_utils import normalize_image

def segment4sdata(
    sdata: sd.SpatialData,
    thresh: float,
    scale: int = 0,
    save: str = "",
    image_key: str = "stain",
    points_key: str = "bin1",  # Stereo-seq bin1 points
) -> sd.SpatialData:
    """Watershed segmentation and aggregate Stereo-seq bin1 to cells."""
    import os
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    # 1. Segment from HE/ssDNA image
    img_dt = sdata.images[image_key]
    img_dt_norm = img_utils.normalize_image(img_dt, compute=True)
    sdata.images[f"{image_key}_norm"] = img_dt_norm
    from .segment import compute_segmentation_thresholds, segment
    img_hires = img_dt_norm[f"scale{scale}"]["image"]
    imgc = sq.im.ImageContainer(img=img_hires)
    thresholds = compute_segmentation_thresholds(imgc["image"], n_classes=3, output_dir=None)
    imgc = segment(img=img_hires, thresh=thresholds[1])

    # 2. Extract mask and add as labels
    seg_mask = imgc["segmented_image"].values.squeeze()
    from spatialdata.models import Labels2DModel
    sdata.labels["cell_labels"] = Labels2DModel.parse(
        seg_mask, 
        transformations=sdata.images[f"{image_key}_norm"].attrs.get("transform", None)
    )
    
    # 3. Convert labels to polygons
    cell_shapes = sd.to_polygons(sdata.labels["cell_labels"])
    sdata.shapes["cells"] = cell_shapes
    
    # 4. Aggregate bin1 points by cell shapes
    # For Stereo-seq: points have 'gene' column, aggregate counts per cell
    aggregated = sd.aggregate(
        values=points_key,
        by="cells",
        values_sdata=sdata,
        by_sdata=sdata,
        value_key="gene",  # categorical gene column in points
        agg_func="count",  # count transcripts per gene per cell
        table_name="cell_expression",
    )
    
    sdata.tables["cell_expression"] = aggregated.tables["cell_expression"]
    
    if save:
        from . import sdata_utils
        sdata_utils.safe_update_sdata(sdata, save)
    return sdata

