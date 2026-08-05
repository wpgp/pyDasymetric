from __future__ import annotations

from doctest import master
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
    geom_path: str
    output_raster_path: str
    id_field: str
    pop_field: str
    mask_path: Optional[str] = None          # optional raster to constrain redistribution
    nibble: Optional[bool] = False           # whether to fill invalid pixels in mastergrid with nearest valid zone id
    nodata: float = -99999.
    block_size: int = 512                    # pixels per side of a processing window
    n_workers: int = field(default_factory=lambda: max(1, (mp.cpu_count() or 2) - 1))
    output_dtype: str = "float32"
    max_blocks: Optional[int] = 256         # for testing, limit number of windows processed


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
            self.pop_df = pd.read_csv(pop_path)
        elif ext in [".shp", ".gpkg", ".geojson"]:
            self.pop_df = pd.DataFrame(gpd.read_file(geom_path))
        else:
            self.pop_df = pd.read_csv(pop_path)

        if self.id_field not in self.pop_df.columns:
            raise ValueError(f"ID field '{id_field}' not found in population data")
            return

        self.pop_df = self.pop_df[[self.id_field, self.pop_field]].copy()

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


def _rasterize_block(
        window: Window,
        vector_path: str,
        id_field: str,
        transform, shape, bounds = None
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

    return (window, zone_raster)


def _nibble_block(
        window: Window,
        mastergrid_path: str, 
        template_path: str):
    """Fill any -1 pixels in zone_raster with the nearest valid zone id,
    constrained to pixels that have valid weight values."""

    master, _, mnodata = _read_by_block(mastergrid_path, window)
    templa, _, tnodata = _read_by_block(template_path, window)

    to_fill = np.logical_or(master != mnodata, templa == tnodata)
    if np.all(to_fill):
        return window,master

    master[master == mnodata] = 0
    filled = fill_nearest(master, to_fill)
    filled[templa == tnodata] = mnodata

    return window,filled

def _apply_mask_block(
        window: Window,
        mastergrid_path: str, 
        mask_path: str):
    """Apply a mask raster to the mastergrid, setting any pixels outside
    the mask to -1 (no zone)."""

    master, _, mnodata = _read_by_block(mastergrid_path, window)
    mask, _, msk_nodata = _read_by_block(mask_path, window)
    master[mask == msk_nodata] = mnodata

    return window,master

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


def _redistribute_block(
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
            gdf = gpd.read_file(self.vector_layer.geom_path)
            if self.config.id_field not in gdf.columns:
                raise ValueError(f"ID field '{self.config.id_field}' not found in geometry data")
            self._create_mastergrid()

        if self.config.nibble:
            self._nibble_mastergrid()

        if self.config.mask_path is not None:
            self._apply_mask()

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
            if self.n_windows < self.config.max_blocks:
                logger.info("Rasterizing mastergrid in a single pass (small raster)...")
                w, zone_raster = _rasterize_block(
                    None,
                    self.vector_layer.geom_path, 
                    self.vector_layer.id_field, 
                    self.weight_source.transform, 
                    (self.weight_source.height, self.weight_source.width)
                    )

                dst.write(zone_raster.astype("int32"), 1)
            else:
                logger.info("Rasterizing mastergrid in parallel (%d windows, %d workers)...", self.n_windows, self.config.n_workers)
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [
                        ex.submit(
                            _rasterize_block,
                            w,
                            self.vector_layer.geom_path,
                            self.vector_layer.id_field,
                            window_transform(w, self.weight_source.transform),
                            (w.height, w.width),
                            window_bounds(w, self.weight_source.transform)
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

        windows = list(self.weight_source.windows())
        out_path = self.config.output_raster_path.replace(".tif", "_mastergrid_nibbled.tif")
        profile = self.weight_source.profile.copy()
        profile.update(dtype="int32", nodata=-1)

        with rasterio.open(out_path, "w", **profile) as dst:
            if self.n_windows < self.config.max_blocks:
                logger.info("Nibbling mastergrid in a single pass (small raster)...")
                w, arr = _nibble_block(
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
                            _nibble_block,
                            w,
                            self.config.geom_path,
                            self.config.weight_raster_path
                        )
                        for w in windows
                    ]
                    for fut in as_completed(futures):
                        w, arr = fut.result()
                        dst.write(arr.astype("int32"), 1, window=w)

        self.config.geom_path = out_path

    def _apply_mask(self) -> None:
        """Apply a mask raster to the mastergrid, setting any pixels outside
        the mask to -1 (no zone)."""
        windows = list(self.weight_source.windows())
        out_path = self.config.output_raster_path.replace(".tif", "_mastergrid_masked.tif")
        profile = self.weight_source.profile.copy()
        profile.update(dtype="int32", nodata=-1)

        with rasterio.open(out_path, "w", **profile) as dst:
            if self.n_windows < self.config.max_blocks:
                logger.info("Applying mask in a single pass (small raster)...")
                w, arr = _apply_mask_block(
                    None, self.config.geom_path, self.config.mask_path
                )
                dst.write(arr.astype("int32"), 1)
            else:
                logger.info("Applying mask in parallel (%d windows, %d workers)...", self.n_windows, self.config.n_workers)
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [
                        ex.submit(
                            _apply_mask_block,
                            w,
                            self.config.geom_path,
                            self.config.mask_path
                        )
                        for w in windows
                    ]
                    for fut in as_completed(futures):
                        w, arr = fut.result()
                        dst.write(arr.astype("int32"), 1, window=w)

        self.config.geom_path = out_path

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

            if self.n_windows < self.config.max_blocks:
                logger.info("Redistributing population in a single pass (small raster)...")
                w, arr = _redistribute_block(
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
                            _redistribute_block,
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
    parser.add_argument("weight_raster", help="Path to the ancillary weight raster (e.g., VIIRS nightlights)")
    parser.add_argument("pop_path", help="Path to the population data file")
    parser.add_argument("geom_path", help="Path to the geometry file")
    parser.add_argument("id_field", help="Field name for the zone IDs in the geometry file")
    parser.add_argument("pop_field", help="Field name for the population data in the population file")
    parser.add_argument("output_raster", help="Path to the output raster file")
    parser.add_argument("--mask-path", default=None, help="Optional raster to constrain redistribution")
    parser.add_argument("--nibble", default=False, help="Fill invalid pixels in mastergrid with nearest valid zone id")
    parser.add_argument("--block-size", type=int, default=512, help="Pixels per side of a processing window")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1), help="Number of parallel worker processes")
    parser.add_argument("--nodata", type=float, default=-1.0, help="No-data value")
    parser.add_argument("--output-dtype", type=str, default="float32", help="Data type for the output raster")
    parser.add_argument("--max-blocks", type=int, default=256, help="Maximum number of blocks to process in parallel")
    args = parser.parse_args()

    config = DasymetricConfig(
        weight_raster_path=args.weight_raster,
        pop_path=args.pop_path,
        geom_path=args.geom_path,
        output_raster_path=args.output_raster,
        id_field=args.id_field,
        pop_field=args.pop_field,
        mask_path=args.mask_path,
        nibble=args.nibble,
        block_size=args.block_size,
        n_workers=args.workers,
        nodata=args.nodata,
        output_dtype=args.output_dtype,
        max_blocks=args.max_blocks,
    )
    
    job = DasymetricRedistributor(config)
    if job is not None:
        job.run()
        logger.info("Verification: %s", job.verify())
    
if __name__ == "__main__":
    _cli()