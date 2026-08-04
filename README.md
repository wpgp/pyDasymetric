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
        weight_raster_path="covariates/slope_weight.tif",
        pop_path="data/county_data.shp",
        geom_path="data/county_data.shp",
        mask_path="masks/urban_mask.tif",  # Masking raster
        pop_field="POP_EST",
        id_field="COUNTY_FP",
        output_raster_path="output/urban_pop_disaggregated.tif",
        n_workers=8
    )

    DasymetricRedistributor(config).run()

Or from the command line:

    python dasymetric.py \\
    path/to/weight_raster.tif \\
    path/to/vector_zones.shp \\
    POP_FIELD \\
    path/to/output_population.tif \\
    --id-field ZONE_ID \\
    --workers 4 \\
    --block-size 512 \\
    --weight-floor 0.0


