from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import threading

from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from utils import fill_nearest

logger = logging.getLogger("dasymetric")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DasymetricConfig:
    weight_raster_path: str
    pop_path: str
    geom_path: str                          # optional mastergrid for zonal stats
    output_raster_path: str
    pop_field: str
    mask_path: Optional[str] = None         # optional raster to constrain redistribution
    id_field: Optional[str] = None          # auto-generated if None
    reference_layer: str = "weight"         # "weight" or "mastergrid" or "vector"
    nibble_mastergrid: bool = False         # whether to fill invalid pixels in mastergrid with nearest valid zone id
    nodata: float = -99999.
    block_size: int = 512                   # pixels per side of a processing window
    n_workers: int = field(default_factory=lambda: max(1, (mp.cpu_count() or 2) - 1))
    output_dtype: str = "float32"
    max_windows: Optional[int] = 256        # for testing, limit number of windows processed


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
class WeightRasterSource:
    """Thin wrapper around the ancillary weight raster: metadata + windowing."""

    def __init__(self, path: str, block_size: int = 512):
        self.path = path
        self.block_size = block_size
        with rasterio.open(path) as src:
            self.profile = src.profile.copy()
            self.transform = src.transform
            self.crs = src.crs
            self.width = src.width
            self.height = src.height
            self.nodata = src.nodata

    def windows(self) -> Iterator[Window]:
        """Yield non-overlapping windows tiling the full raster extent."""
        for row_off in range(0, self.height, self.block_size):
            h = min(self.block_size, self.height - row_off)
            for col_off in range(0, self.width, self.block_size):
                w = min(self.block_size, self.width - col_off)
                yield Window(col_off, row_off, w, h)


class VectorPopulationLayer:
    """Wraps the admin-unit vector layer and its population attribute."""

    def __init__(self, 
                 pop_path: str, 
                 geom_path: str, 
                 pop_field: str, 
                 id_field: Optional[str] = '_zone_id'
                 ):
        self.pop_path = pop_path
        self.geom_path = geom_path
        self.pop_field = pop_field
        self.id_field = id_field
        self.has_mastergrid = False

        ext = Path(geom_path).suffix.lower()
        if ext in [".tif"]:
            self.has_mastergrid = True
            self.pop_df = pd.read_csv(pop_path)[[id_field, pop_field]]
        elif ext in [".shp", ".gpkg", ".geojson"]:
            self.pop_df = pd.DataFrame(gpd.read_file(geom_path)[[id_field, pop_field]])
        else:
            self.pop_df = pd.read_csv(pop_path)

        # zone id -1 is reserved to mean "no zone" during rasterisation
        if (self.pop_df[id_field] == -1).any():
            raise ValueError("zone id -1 is reserved; re-map ids so none are -1")

    def total_population(self) -> float:
        return float(self.pop_df[self.pop_field].sum())

    def materialize(self, out_path: str) -> str:
        """Persist the (possibly reprojected) layer so worker processes can
        each open their own independent, bbox-filterable copy."""
        self.pop_df.to_csv(out_path, index=False)
        return out_path


# --------------------------------------------------------------------------- #
# Worker functions (module-level so they are picklable for multiprocessing)
# --------------------------------------------------------------------------- #
def _read_by_block(path: str, window: Window):
    with rasterio.open(path) as src:
        block = src.read(1, window=window).astype("float64")
        if window is not None:
            transform = src.window_transform(window)
        else:
            transform = src.transform
        nodata = src.nodata
    return block, transform, nodata


def _valid_mask(weight: np.ndarray, weight_nodata, zone_raster: np.ndarray, weight_floor: float) -> np.ndarray:
    valid = np.isfinite(weight)
    if weight_nodata is not None:
        valid &= weight != weight_nodata
    valid &= weight > weight_floor
    valid &= zone_raster != -1

    return valid


def _rasterize_zones(
        window: Window,
        vector_path: str,
        id_field: str,
        transform, shape, bounds = None,
        mask_path: Optional[str] = None
        ):
    """Rasterize the features intersecting this window's bounds, optionally
    applying a mask to constrain the output."""
    gdf = gpd.read_file(vector_path, bbox=bounds)
    if gdf.empty:
        return window, np.full(shape, -1, dtype="int32")
    
    shapes = list(zip(gdf.geometry, gdf[id_field].astype("int32")))
    zone_raster = rasterize(
        shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32"
    )
    if mask_path is not None:
        with rasterio.open(mask_path) as src:
            mask = src.read(1, window=window).astype("bool")
        zone_raster[~mask] = -1

    return (window, zone_raster)


def nibble_zones(
        window: Window,
        mastergrid_path: str, 
        template_path: str):
    """Fill any -1 pixels in zone_raster with the nearest valid zone id,
    constrained to pixels that have valid weight values."""

    master, _, mnodata = _read_by_block(mastergrid_path, window)
    templa, _, tnodata = _read_by_block(template_path, window)

    mask = np.logical_and(master == mnodata, templa == tnodata)
    if not np.any(mask):
        return master

    filled = fill_nearest(master, mask)
    return filled

def _compute_partial_zone_sums(
    window: Window,
    weight_path: str,
    mastergrid_path: str
):
    """Pass 1 worker: sum of weight pixels per zone, within one window."""
    weight, _, wnodata = _read_by_block(weight_path, window)
    master, _, mnodata = _read_by_block(mastergrid_path, window)
    
    valid = np.logical_and(weight != wnodata, master != mnodata)
    
    df = pd.DataFrame()
    df['zones'] = master[valid].flatten()
    df['weights'] = weight[valid].flatten()
    agg = df.groupby('zones')['weights'].sum().reset_index()
    
    return agg


def _redistribute_window(
    window: Window,
    weight_path: str,
    mastergrid_path: str,
    zone_sums: pd.DataFrame,
    id_field: str,
    nodata: float,
    out_dtype: str,
):
    """Pass 2 worker: convert one window's weights into population estimates."""
    weight, _, wnodata = _read_by_block(weight_path, window)
    master, _, mnodata = _read_by_block(mastergrid_path, window)

    lookup_dict = zone_sums.set_index(id_field)["norm"].to_dict()
    updater = np.vectorize(lambda x: lookup_dict.get(x, 0.0), otypes=[np.float32])
    norm = updater(master)

    masked = np.logical_or(weight == wnodata, master == mnodata)
    out = weight * norm
    out[~np.isfinite(norm)] = nodata
    out = out.astype(out_dtype)
    out[masked] = nodata

    return window, out

# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class DasymetricRedistributor:
    """Coordinates the two-pass, parallel dasymetric redistribution."""

    def __init__(self, config: DasymetricConfig):
        self.config = config

        self.weight_source = WeightRasterSource(config.weight_raster_path, config.block_size)
        self.vector_layer = VectorPopulationLayer(
            config.pop_path, config.geom_path, config.pop_field, config.id_field
        )

        self.n_windows = len(list(self.weight_source.windows()))

        if not self.vector_layer.has_mastergrid:
            self._create_mastergrid()

        if self.config.nibble_mastergrid:
            self._nibble_mastergrid()

        self._zone_sums = None

    # -- public API --------------------------------------------------------
    def run(self) -> "DasymetricRedistributor":
        zone_sums = self._compute_zone_sums()
        zone_sums = pd.merge(zone_sums,
            self.vector_layer.pop_df,
            on=self.config.id_field, how='left')
        zone_sums['norm'] = np.divide(
            zone_sums[self.config.pop_field].values,
            zone_sums['weights'].values, 
            out=np.zeros_like(zone_sums['weights'].values),
            where=zone_sums['weights'].values!=0)
        self._zone_sums = zone_sums.copy()

        self._redistribute()

        logger.info("Done. Output written to %s", self.config.output_raster_path)

    def verify(self) -> Dict[str, float]:
        """Sanity check: total input population vs. total output population.
        Small discrepancies are expected for zones with zero total weight
        (no plausible pixels to redistribute into)."""
        input_total = self.vector_layer.total_population()
        output_total = 0.0
        with rasterio.open(self.config.output_raster_path) as src:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                block = block[block != src.nodata]
                output_total += float(block.sum())
        diff = input_total - output_total
        pct = (diff / input_total * 100) if input_total else 0.0
        return {
            "input_population": input_total,
            "output_population": output_total,
            "difference": diff,
            "difference_pct": pct,
        }

    # -- internal ------------------------------------------------------------
    def _create_mastergrid(self) -> None:
        """If the user did not provide a mastergrid, create one by rasterizing
        the vector layer into the weight raster's CRS and extent."""
        if self.vector_layer.has_mastergrid:
            return

        logger.info("Creating mastergrid from vector layer...")
        windows = list(self.weight_source.windows())
        out_path = self.config.output_raster_path.replace(".tif", "_mastergrid.tif")
        self.config.geom_path = out_path
        profile = self.weight_source.profile.copy()
        profile.update(dtype="int32", nodata=-1)

        with rasterio.open(out_path, "w", **profile) as dst:
            if self.n_windows < self.config.max_windows:
                logger.info("Rasterizing mastergrid in a single pass (small raster)...")
                w, zone_raster = _rasterize_zones(
                    None,
                    self.vector_layer.geom_path, 
                    self.vector_layer.id_field, 
                    self.weight_source.transform, 
                    (self.weight_source.height, self.weight_source.width),
                    mask_path=self.config.mask_path
                    )

                dst.write(zone_raster.astype("int32"), 1)
            else:
                logger.info("Rasterizing mastergrid in parallel (%d windows, %d workers)...", self.n_windows, self.config.n_workers)
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [
                        ex.submit(
                            _rasterize_zones,
                            w,
                            self.vector_layer.geom_path,
                            self.vector_layer.id_field,
                            window_transform(w, self.weight_source.transform),
                            (w.height, w.width),
                            window_bounds(w, self.weight_source.transform),
                            self.config.mask_path
                        )
                        for w in windows
                    ]
                    self.futures = futures

                    for fut in as_completed(futures):
                        w, zone_raster = fut.result()
                        dst.write(zone_raster.astype("int32"), 1, window=w)

    def _nibble_mastergrid(self) -> None:
        """Fill any -1 pixels in the mastergrid with the nearest valid zone id,
        constrained to pixels that have valid weight values."""
        if not self.vector_layer.has_mastergrid:
            return

        logger.info("Nibbling mastergrid to fill gaps...")
        windows = list(self.weight_source.windows())
        out_path = self.config.output_raster_path.replace(".tif", "_mastergrid_nibbled.tif")
        self.config.geom_path = out_path
        profile = self.weight_source.profile.copy()
        profile.update(dtype="int32", nodata=-1)

        with rasterio.open(out_path, "w", **profile) as dst:
            if self.n_windows < self.config.max_windows:
                logger.info("Nibbling mastergrid in a single pass (small raster)...")
                w, arr = nibble_zones(
                    None,
                    self.config.geom_path,
                    self.config.weight_raster_path
                )
                dst.write(arr.astype("int32"), 1)
            else:
                logger.info("Nibbling mastergrid in parallel (%d windows, %d workers)...", self.n_windows, self.config.n_workers)
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [
                        ex.submit(
                            nibble_zones,
                            w,
                            self.config.geom_path,
                            self.config.weight_raster_path
                        )
                        for w in windows
                    ]
                    for fut in as_completed(futures):
                        w, arr = fut.result()
                        dst.write(arr.astype("int32"), 1, window=w)

    def _compute_zone_sums(self):
        windows = list(self.weight_source.windows())
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
            futures = [
                ex.submit(
                    _compute_partial_zone_sums,
                    w,
                    self.config.weight_raster_path,
                    self.config.geom_path,
                )
                for w in windows
            ]

            totals = []

            for fut in as_completed(futures):
                totals.append(fut.result())

            stack = pd.concat(totals, ignore_index=True)
            stack.rename(columns={'zones': self.config.id_field}, inplace=True)
            stack = stack.groupby(self.config.id_field)['weights'].sum().reset_index()
        return stack

    def _redistribute(self):
        profile = self.weight_source.profile.copy()
        profile.update(
            dtype=self.config.output_dtype,
            count=1,
            nodata=self.config.nodata,
            blockxsize=min(512, self.config.block_size),
            blockysize=min(512, self.config.block_size),
            BIGTIFF="IF_SAFER",
        )
        windows = list(self.weight_source.windows())

        with rasterio.open(self.config.output_raster_path, "w", **profile) as dst:

            if self.n_windows < self.config.max_windows:
                logger.info("Redistributing population in a single pass (small raster)...")
                w, arr = _redistribute_window(
                    None,
                    self.config.weight_raster_path,
                    self.config.geom_path,
                    self._zone_sums,
                    self.config.id_field,
                    self.config.nodata,
                    self.config.output_dtype,
                )
                dst.write(arr, 1)
            else:
                logger.info("Redistributing population in parallel (%d windows, %d workers)...", self.n_windows, self.config.n_workers)
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [
                        ex.submit(
                            _redistribute_window,
                            w,
                            self.config.weight_raster_path,
                            self.config.geom_path,
                            self._zone_sums,
                            self.config.id_field,
                            self.config.nodata,
                            self.config.output_dtype,
                        )
                        for w in windows
                    ]
                    # Only the main process writes -> no concurrent-write issues.
                    for fut in as_completed(futures):
                        w, arr = fut.result()
                        dst.write(arr, 1, window=w)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dasymetric population redistribution")
    parser.add_argument("weight_raster")
    parser.add_argument("pop_path")
    parser.add_argument("geom_path")
    parser.add_argument("pop_field")
    parser.add_argument("output_raster")
    parser.add_argument("--mask-path", default=None)    
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--nodata", type=float, default=-1.0)
    parser.add_argument("--weight-floor", type=float, default=0.0)
    args = parser.parse_args()

    config = DasymetricConfig(
        weight_raster_path=args.weight_raster,
        pop_path=args.pop_path,
        geom_path=args.geom_path,
        output_raster_path=args.output_raster,
        pop_field=args.pop_field,
        id_field=args.id_field,
        mask_path=args.mask_path,
        block_size=args.block_size,
        n_workers=args.workers,
        nodata=args.nodata,
        weight_floor=args.weight_floor,
    )
    job = DasymetricRedistributor(config).run()
    logger.info("Verification: %s", job.verify())


if __name__ == "__main__":
    _cli()