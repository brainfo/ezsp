"""Reusable utilities for generating, validating, and persisting spatial crops.

Works with both SpatialData and AnnData objects.  Crops are stored as
human-readable JSON so they can be inspected and shared across workflows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial extent
# ---------------------------------------------------------------------------

def get_spatial_extent(
    data,
    shapes_key: str | None = None,
    spatial_key: str = "spatial",
) -> tuple[float, float, float, float] | None:
    """Return ``(xmin, ymin, xmax, ymax)`` for *data*.

    Parameters
    ----------
    data
        A ``SpatialData`` or ``AnnData`` object.
    shapes_key
        For SpatialData: name of the shapes element to read bounds from.
        ``None`` picks the first available shape.
    spatial_key
        For AnnData: key in ``obsm`` that stores coordinates.
    """
    # SpatialData path
    if hasattr(data, "shapes"):
        keys = [shapes_key] if shapes_key else list(data.shapes.keys())
        for key in keys:
            if key in data.shapes:
                bounds = data.shapes[key].total_bounds  # [minx, miny, maxx, maxy]
                return (bounds[0], bounds[1], bounds[2], bounds[3])
        # Fallback: try table coordinates
        for table in data.tables.values():
            if spatial_key in table.obsm:
                coords = table.obsm[spatial_key]
                return (
                    coords[:, 0].min(),
                    coords[:, 1].min(),
                    coords[:, 0].max(),
                    coords[:, 1].max(),
                )
        return None

    # AnnData path
    if hasattr(data, "obsm") and spatial_key in data.obsm:
        coords = data.obsm[spatial_key]
        return (
            coords[:, 0].min(),
            coords[:, 1].min(),
            coords[:, 0].max(),
            coords[:, 1].max(),
        )

    return None


# ---------------------------------------------------------------------------
# Crop generation
# ---------------------------------------------------------------------------

def generate_random_crops(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crop_size: float,
    n_crops: int,
    seed: int,
    *,
    as_int: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Generate *n_crops* random bounding boxes within the given extent.

    Parameters
    ----------
    xmin, ymin, xmax, ymax
        Outer spatial bounds.
    crop_size
        Side length of each square crop.
    n_crops
        Number of crops to generate.
    seed
        Random seed for reproducibility.
    as_int
        If ``True`` cast coordinates to ``int`` (useful for pixel-based
        workflows such as squidpy).

    Returns
    -------
    list of (x1, y1, x2, y2) tuples.
    """
    rng = np.random.default_rng(seed)

    valid_xmax = xmax - crop_size
    valid_ymax = ymax - crop_size

    if valid_xmax <= xmin or valid_ymax <= ymin:
        logger.warning("Spatial extent too small for crop_size=%s", crop_size)
        return []

    crops: list[tuple] = []
    for _ in range(n_crops):
        x1 = rng.uniform(xmin, valid_xmax)
        y1 = rng.uniform(ymin, valid_ymax)
        if as_int:
            ix1, iy1 = int(x1), int(y1)
            crops.append((ix1, iy1, ix1 + int(crop_size), iy1 + int(crop_size)))
        else:
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            crops.append((x1, y1, x2, y2))

    return crops


# ---------------------------------------------------------------------------
# Crop validation
# ---------------------------------------------------------------------------

def validate_crops(
    crops: list[tuple[float, float, float, float]],
    adata,
    *,
    min_cells: int = 10,
    spatial_key: str = "spatial",
    verbose: bool = True,
) -> list[tuple[float, float, float, float]]:
    """Keep only crops that lie within data bounds and contain enough cells.

    Parameters
    ----------
    crops
        List of ``(x1, y1, x2, y2)`` tuples.
    adata
        An ``AnnData`` with coordinates in ``obsm[spatial_key]``.
    min_cells
        Minimum number of cells/bins required inside the crop.
    spatial_key
        Key in ``adata.obsm`` that holds spatial coordinates.
    verbose
        Print diagnostics for discarded crops.
    """
    if spatial_key not in adata.obsm:
        logger.warning(
            "spatial_key=%r not found in adata.obsm; skipping validation", spatial_key,
        )
        return list(crops)

    coords = adata.obsm[spatial_key]
    data_xmin, data_ymin = coords.min(axis=0)[:2]
    data_xmax, data_ymax = coords.max(axis=0)[:2]

    valid: list[tuple] = []
    for crop in crops:
        x1, y1, x2, y2 = crop

        if not (x1 >= data_xmin and y1 >= data_ymin and x2 <= data_xmax and y2 <= data_ymax):
            if verbose:
                logger.info(
                    "Discarding crop %s: outside data bounds "
                    "[%.0f-%.0f, %.0f-%.0f]",
                    crop, data_xmin, data_xmax, data_ymin, data_ymax,
                )
            continue

        mask = (
            (coords[:, 0] >= x1) & (coords[:, 0] <= x2)
            & (coords[:, 1] >= y1) & (coords[:, 1] <= y2)
        )
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            if verbose:
                logger.info(
                    "Discarding crop %s: only %d cells (min=%d)",
                    crop, n_cells, min_cells,
                )
            continue

        valid.append(crop)

    return valid


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_crops(
    crops: list[tuple[float, float, float, float]],
    filepath: str | Path,
    *,
    sample_name: str,
    crop_size: float,
    unit: str = "micrometers",
    seed: int,
    n_requested: int | None = None,
    spatial_extent: tuple[float, float, float, float] | None = None,
    metadata: dict[str, Any] | None = None,
    sdata=None,
    crop_shapes_key: str = "crop_boxes",
) -> Path:
    """Save crop coordinates and generation parameters to a JSON file.

    When *sdata* is provided the crops are also stored as box polygons in
    ``sdata.shapes[crop_shapes_key]`` and persisted with
    ``sdata.write_element(crop_shapes_key)``.

    Returns the resolved output path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        "sample_name": sample_name,
        "crop_size": crop_size,
        "unit": unit,
        "seed": seed,
        "n_requested": n_requested if n_requested is not None else len(crops),
        "n_crops": len(crops),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "spatial_extent": list(spatial_extent) if spatial_extent is not None else None,
        "crops": [list(c) for c in crops],
        "metadata": metadata or {},
    }

    with open(filepath, "w") as fh:
        json.dump(doc, fh, indent=2)

    # Persist as SpatialData shapes element
    if sdata is not None and crops:
        try:
            from shapely import box
            import geopandas as gpd
            from spatialdata.models import ShapesModel

            gdf = gpd.GeoDataFrame(
                {"crop_id": [f"crop_{i}" for i in range(len(crops))]},
                geometry=[box(x1, y1, x2, y2) for x1, y1, x2, y2 in crops],
            )
            gdf = ShapesModel.parse(gdf)
            sdata.shapes[crop_shapes_key] = gdf
            if sdata.is_backed():
                sdata.write_element(crop_shapes_key, overwrite=True)
                logger.info("Wrote shapes element %r to backing store", crop_shapes_key)
        except Exception:
            logger.exception("Failed to write crop shapes to SpatialData")

    return filepath


def load_crops(filepath: str | Path) -> dict[str, Any]:
    """Load a previously saved crop file.

    Returns a dict whose ``"crops"`` value is a list of tuples ready for use.
    All other generation parameters are included as-is.
    """
    filepath = Path(filepath)
    with open(filepath) as fh:
        doc = json.load(fh)

    doc["crops"] = [tuple(c) for c in doc["crops"]]
    if doc.get("spatial_extent") is not None:
        doc["spatial_extent"] = tuple(doc["spatial_extent"])
    return doc


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def generate_and_save_crops(
    data,
    output_path: str | Path,
    *,
    sample_name: str,
    crop_size: float,
    n_crops: int,
    seed: int,
    unit: str = "micrometers",
    min_cells: int = 10,
    validate: bool = True,
    as_int: bool = False,
    shapes_key: str | None = None,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    crop_shapes_key: str = "crop_boxes",
) -> list[tuple[float, float, float, float]]:
    """Generate crops, optionally validate them, save to JSON, and return.

    When *data* is a ``SpatialData`` object the crops are additionally stored
    as box polygons in ``sdata.shapes[crop_shapes_key]`` and written to the
    backing store.

    Parameters
    ----------
    data
        ``SpatialData`` or ``AnnData``.
    output_path
        Destination JSON file.
    sample_name
        Label stored in the JSON (e.g. ``"PLA330"``).
    crop_size
        Side length of each square crop.
    n_crops
        Number of crops to request.
    seed
        Random seed.
    unit
        Unit label stored in the JSON (e.g. ``"micrometers"`` or ``"pixels"``).
    min_cells
        Minimum cells per crop when *validate* is ``True``.
    validate
        If ``True``, discard crops that fail :func:`validate_crops`.
    as_int
        Cast coordinates to ``int`` (pixel-based workflows).
    shapes_key
        Passed to :func:`get_spatial_extent`.
    spatial_key
        Key in ``obsm`` for coordinates.
    table_key
        For SpatialData: name of the table to validate against.
        ``None`` searches all tables for one containing *spatial_key*.
    metadata
        Arbitrary extra info stored in the JSON.
    crop_shapes_key
        Name of the shapes element stored in SpatialData (default
        ``"crop_boxes"``).  Ignored when *data* is an ``AnnData``.
    """
    extent = get_spatial_extent(data, shapes_key=shapes_key, spatial_key=spatial_key)
    if extent is None:
        raise ValueError("Could not determine spatial extent from the supplied data.")

    xmin, ymin, xmax, ymax = extent
    crops = generate_random_crops(
        xmin, ymin, xmax, ymax,
        crop_size=crop_size,
        n_crops=n_crops,
        seed=seed,
        as_int=as_int,
    )

    if validate:
        # Resolve the AnnData to validate against
        if hasattr(data, "obsm"):
            adata = data
        else:
            # SpatialData -- resolve table by key or search all tables
            adata = None
            if table_key is not None:
                tbl = data.tables.get(table_key)
                if tbl is not None and spatial_key in tbl.obsm:
                    adata = tbl
                else:
                    logger.warning(
                        "table_key=%r not found or missing spatial_key=%r",
                        table_key, spatial_key,
                    )
            else:
                for name, tbl in data.tables.items():
                    if spatial_key in tbl.obsm:
                        adata = tbl
                        logger.info("Using table %r for crop validation", name)
                        break

        if adata is not None:
            crops = validate_crops(
                crops, adata, min_cells=min_cells, spatial_key=spatial_key,
            )

    # Pass sdata only when data is a SpatialData object (has .shapes attr)
    _sdata = data if hasattr(data, "shapes") else None

    save_crops(
        crops,
        output_path,
        sample_name=sample_name,
        crop_size=crop_size,
        unit=unit,
        seed=seed,
        n_requested=n_crops,
        spatial_extent=extent,
        metadata=metadata,
        sdata=_sdata,
        crop_shapes_key=crop_shapes_key,
    )

    return crops
