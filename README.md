# pyDasymetric
To perform dasymetric redistribution of population.

"""
Given:
  * A raster of relative "weights" (probability / suitability / density proxy
    for human presence — e.g. built-up area fraction, night-lights, land cover
    likelihood surface).
  * A population layer containing total population count per administrative unit.
  * A geometry layer describing the administrative unit boundary.

This produces a raster where each pixel holds an estimate of population,
computed as:

    pop(pixel) = pop(zone) * weight(pixel) / sum(weight(pixels in zone))

The algorithm runs in two windowed passes over the raster so that files far
larger than RAM can be processed:

  Pass 1 (parallel): for every raster block, rasterise the admin polygons
      that intersect it and accumulate a partial sum of weights per zone.
      Partial sums are combined in the main process into exact global sums.

  Pass 2 (parallel): for every raster block, rasterise again, look up each
      zone's global weight sum and population, and redistribute population
      into a per-pixel output array. Results are streamed back to the main
      process, which is the only process writing to the output GeoTIFF.

Usage
-----
    from dasymetric import DasymetricConfig, DasymetricRedistributor

    config = DasymetricConfig(
        weight_raster_path="covariates/viirs_2024.tif",
        pop_path="data/pop_count.csv",
        geom_path="data/adm_boundary.shp",
        mask_path="masks/settlement_mask.tif",  # Masking raster
        pop_field="POP_EST",
        id_field="COUNTY_FP",
        nibble=True,
        output_raster_path="out/urban_pop_disaggregated.tif",
        n_workers=8,
        max_blocks=128
        block_size=512
    )

    DasymetricRedistributor(config).run()

Or from the command line:

    python dasymetric.py \\
    path/to/weight_raster.tif \\
    path/to/mastergrid.tif \\
    path/to/pop_count.csv \\
    ID_FIELD \\
    POP_FIELD \\
    path/to/output_population.tif \\
    --workers 4 \\
    --block-size 512


