# pyDasymetric
To perform dasymetric redistribution of population.

"""
Given:
  * A raster of relative "weights" (probability / suitability / density proxy
    for human presence — e.g. built-up area fraction, night-lights, land cover
    likelihood surface).
  * A vector layer of administrative units, each carrying a total population
    count.

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
        weight_raster_path="weight.tif",
        vector_path="admin_units.gpkg",
        output_raster_path="population_dasymetric.tif",
        pop_field="population",
        id_field="admin_id",     # optional; auto-generated if omitted
        block_size=1024,
        n_workers=8,
    )
    job = DasymetricRedistributor(config)
    job.run()
    report = job.verify()
    print(report)

Or from the command line:

    python dasymetric.py weight.tif admin_units.gpkg pop_field out.tif \
        --id-field admin_id --block-size 1024 --workers 8
"""

