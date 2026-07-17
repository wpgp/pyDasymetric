from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds

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
    vector_path: str
    output_raster_path: str
    pop_field: str
    id_field: Optional[str] = None          # auto-generated if None
    nodata: float = -1.0
    block_size: int = 512                   # pixels per side of a processing window
    n_workers: int = field(default_factory=lambda: max(1, (mp.cpu_count() or 2) - 1))
    output_dtype: str = "float32"
    weight_floor: float = 0.0               # pixels with weight <= this are excluded


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

    def __init__(self, path: str, pop_field: str, id_field: Optional[str] = None):
        self.path = path
        self.pop_field = pop_field
        self.gdf = gpd.read_file(path)

        if self.pop_field not in self.gdf.columns:
            raise ValueError(f"'{pop_field}' not found in vector attributes")

        if id_field is None:
            self.id_field = "_zone_id"
            self.gdf[self.id_field] = np.arange(1, len(self.gdf) + 1, dtype=np.int32)
        else:
            if id_field not in self.gdf.columns:
                raise ValueError(f"'{id_field}' not found in vector attributes")
            self.id_field = id_field

        # zone id 0 is reserved to mean "no zone" during rasterisation
        if (self.gdf[self.id_field] == 0).any():
            raise ValueError("zone id 0 is reserved; re-map ids so none are 0")

    def reproject_to(self, crs) -> "VectorPopulationLayer":
        if self.gdf.crs != crs:
            self.gdf = self.gdf.to_crs(crs)
        return self

    def total_population(self) -> float:
        return float(self.gdf[self.pop_field].sum())

    def materialize(self, out_path: str) -> str:
        """Persist the (possibly reprojected) layer so worker processes can
        each open their own independent, bbox-filterable copy."""
        self.gdf.to_file(out_path, driver="GPKG")
        return out_path


# --------------------------------------------------------------------------- #
# Worker functions (module-level so they are picklable for multiprocessing)
# --------------------------------------------------------------------------- #
def _read_weight_block(weight_path: str, window: Window):
    with rasterio.open(weight_path) as src:
        block = src.read(1, window=window).astype("float64")
        transform = src.window_transform(window)
        nodata = src.nodata
    return block, transform, nodata


def _rasterize_zones(vector_path: str, id_field: str, transform, shape, bounds):
    """Read only the features intersecting this window's bounds and burn
    their zone id into an integer raster aligned with the weight block."""
    gdf = gpd.read_file(vector_path, bbox=bounds)
    if gdf.empty:
        return None, gdf
    shapes = list(zip(gdf.geometry, gdf[id_field].astype("int32")))
    zone_raster = rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, dtype="int32"
    )
    return zone_raster, gdf


def _valid_mask(weight: np.ndarray, weight_nodata, zone_raster: np.ndarray, weight_floor: float):
    valid = np.isfinite(weight)
    if weight_nodata is not None:
        valid &= weight != weight_nodata
    valid &= weight > weight_floor
    valid &= zone_raster != 0
    return valid


def _compute_partial_zone_sums(
    window: Window,
    weight_path: str,
    vector_path: str,
    id_field: str,
    weight_floor: float,
) -> Dict[int, float]:
    """Pass 1 worker: sum of weight pixels per zone, within one window."""
    weight, transform, wnodata = _read_weight_block(weight_path, window)
    b = window_bounds(window, transform)
    zone_raster, _ = _rasterize_zones(vector_path, id_field, transform, weight.shape, b)
    if zone_raster is None:
        return {}

    valid = _valid_mask(weight, wnodata, zone_raster, weight_floor)
    if not valid.any():
        return {}

    zones = zone_raster[valid]
    vals = weight[valid]
    order = np.argsort(zones)
    zones_sorted = zones[order]
    vals_sorted = vals[order]
    uniq, start_idx = np.unique(zones_sorted, return_index=True)
    sums = np.add.reduceat(vals_sorted, start_idx)
    return {int(z): float(s) for z, s in zip(uniq, sums)}


def _redistribute_window(
    window: Window,
    weight_path: str,
    vector_path: str,
    id_field: str,
    pop_field: str,
    zone_sums: Dict[int, float],
    nodata: float,
    weight_floor: float,
    out_dtype: str,
) -> Tuple[Window, np.ndarray]:
    """Pass 2 worker: convert one window's weights into population estimates."""
    weight, transform, wnodata = _read_weight_block(weight_path, window)
    out = np.full(weight.shape, nodata, dtype=out_dtype)

    b = window_bounds(window, transform)
    zone_raster, gdf = _rasterize_zones(vector_path, id_field, transform, weight.shape, b)
    if zone_raster is None:
        return window, out

    valid = _valid_mask(weight, wnodata, zone_raster, weight_floor)
    if not valid.any():
        return window, out

    pop_lookup = dict(zip(gdf[id_field].astype("int32"), gdf[pop_field].astype("float64")))
    result = np.zeros(weight.shape, dtype="float64")

    for z in np.unique(zone_raster[valid]):
        z = int(z)
        zsum = zone_sums.get(z, 0.0)
        pop = pop_lookup.get(z, 0.0)
        if zsum <= 0 or pop <= 0:
            continue
        mask = valid & (zone_raster == z)
        result[mask] = weight[mask] / zsum * pop

    out[valid] = result[valid].astype(out_dtype)
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
            config.vector_path, config.pop_field, config.id_field
        ).reproject_to(self.weight_source.crs)

        self._tmp_vector_path = str(
            Path(config.output_raster_path).with_suffix("").as_posix() + "_zones_tmp.gpkg"
        )
        self.vector_layer.materialize(self._tmp_vector_path)
        self._zone_sums: Optional[Dict[int, float]] = None

    # -- public API --------------------------------------------------------
    def run(self) -> "DasymetricRedistributor":
        logger.info("Pass 1/2: computing per-zone weight sums (%d workers)", self.config.n_workers)
        self._zone_sums = self._compute_zone_sums()

        logger.info("Pass 2/2: redistributing population (%d workers)", self.config.n_workers)
        self._write_output(self._zone_sums)

        self._cleanup()
        logger.info("Done. Output written to %s", self.config.output_raster_path)
        return self

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
    def _compute_zone_sums(self) -> Dict[int, float]:
        windows = list(self.weight_source.windows())
        totals: Dict[int, float] = {}
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
            futures = [
                ex.submit(
                    _compute_partial_zone_sums,
                    w,
                    self.config.weight_raster_path,
                    self._tmp_vector_path,
                    self.vector_layer.id_field,
                    self.config.weight_floor,
                )
                for w in windows
            ]
            for fut in as_completed(futures):
                for z, s in fut.result().items():
                    totals[z] = totals.get(z, 0.0) + s
        return totals

    def _write_output(self, zone_sums: Dict[int, float]) -> None:
        profile = self.weight_source.profile.copy()
        profile.update(
            dtype=self.config.output_dtype,
            count=1,
            nodata=self.config.nodata,
            compress="lzw",
            tiled=True,
            blockxsize=min(256, self.config.block_size),
            blockysize=min(256, self.config.block_size),
            BIGTIFF="IF_SAFER",
        )
        windows = list(self.weight_source.windows())

        with rasterio.open(self.config.output_raster_path, "w", **profile) as dst:
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                futures = [
                    ex.submit(
                        _redistribute_window,
                        w,
                        self.config.weight_raster_path,
                        self._tmp_vector_path,
                        self.vector_layer.id_field,
                        self.config.pop_field,
                        zone_sums,
                        self.config.nodata,
                        self.config.weight_floor,
                        self.config.output_dtype,
                    )
                    for w in windows
                ]
                # Only the main process writes -> no concurrent-write issues.
                for fut in as_completed(futures):
                    window, arr = fut.result()
                    dst.write(arr, 1, window=window)

    def _cleanup(self) -> None:
        p = Path(self._tmp_vector_path)
        for sibling in p.parent.glob(p.stem + ".*"):
            try:
                sibling.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dasymetric population redistribution")
    parser.add_argument("weight_raster")
    parser.add_argument("vector_path")
    parser.add_argument("pop_field")
    parser.add_argument("output_raster")
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--nodata", type=float, default=-1.0)
    parser.add_argument("--weight-floor", type=float, default=0.0)
    args = parser.parse_args()

    config = DasymetricConfig(
        weight_raster_path=args.weight_raster,
        vector_path=args.vector_path,
        output_raster_path=args.output_raster,
        pop_field=args.pop_field,
        id_field=args.id_field,
        block_size=args.block_size,
        n_workers=args.workers,
        nodata=args.nodata,
        weight_floor=args.weight_floor,
    )
    job = DasymetricRedistributor(config).run()
    logger.info("Verification: %s", job.verify())


#if __name__ == "__main__":
#    _cli()