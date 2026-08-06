[![DOI](https://zenodo.org/badge/1298383082.svg)](https://doi.org/10.5281/zenodo.21815904)

# pyDasymetric
![Illustration](fig/illustration.png)

To perform dasymetric redistribution of population.

Given:
  * A raster of relative "weights" (probability / suitability / density proxy     for human presence — e.g. built-up area fraction, night-lights, land cover likelihood surface).
  * A population layer containing total population count per administrative unit.
  * A geometry layer describing the administrative unit boundary. This can be in the form of shapefile or raster defining "id" for each administrative unit. In either input the "id" should be synchronised with the one in the population layer.

This produces a raster where each pixel holds an estimate of population, computed as:

    pop(pixel) = pop(zone) * weight(pixel) / sum(weight(pixels in zone))

If the raster has more blocks/windows than `max_blocks`, the algorithm runs in two windowed passes over the raster. In this way files that are larger than RAM can be processed:

  * Pass 1 (parallel): for every raster block, rasterise the admin polygons that intersect it and accumulate a partial sum of weights per zone. Partial sums are combined in the main process into exact global sums.

  * Pass 2 (parallel): for every raster block, rasterise again, look up each zone's global weight sum and population, and redistribute population into a per-pixel output array. Results are streamed back to the main process, which is the only process writing to the output GeoTIFF.

Usage
-----
This tool can be used in command line:

    dasymetric.py [-h] [--mask-path MASK_PATH] 
                    [--nibble NIBBLE] [--block-size BLOCK_SIZE] 
                    [--workers WORKERS]
                    [--nodata NODATA] [--output-dtype OUTPUT_DTYPE] 
                    [--max-blocks MAX_BLOCKS]
                    weight_raster pop_path geom_path 
                    id_field pop_field output_raster

    Dasymetric population redistribution

    positional arguments:
    weight_raster         Path to the ancillary weight raster
    pop_path              Path to the population data file
    geom_path             Path to the geometry file
    id_field              Field name for the zone IDs in the geometry file
    pop_field             Field name for the population data in the 
                          population file
    output_raster         Path to the output raster file

    options:
    -h, --help             show this help message and exit
    --mask-path MASK_PATH  Optional raster to constrain redistribution
    --nibble NIBBLE        Fill invalid pixels in mastergrid with nearest 
                           valid zone id
    --block-size BLOCK_SIZE  Pixels per side of a processing window
    --workers WORKERS      Number of parallel worker processes
    --nodata NODATA        No-data value
    --output-dtype OUTPUT_DTYPE  Data type for the output raster
    --max-blocks MAX_BLOCKS  Maximum number of blocks to process in parallel

or as a code block:

    from dasymetric import DasymetricConfig, DasymetricRedistributor

    config = DasymetricConfig(
        weight_raster_path="covariates/weight_raster.tif", #required
        pop_path="data/pop_count.csv",                  #required
        geom_path="data/adm_boundary.shp",              #required 
        pop_field="POP_FIELD",                          #required
        id_field="ID_FIELD",                            #required
        output_raster_path="out/pop_disaggregated.tif", #required
        nibble=False,                                   #optional
        n_workers=8,                                    #optional
        max_blocks=128,                                 #optional
        block_size=512,                                 #optional
        nodata=-99,                                     #optional
        output_dtype='float64'                          #optional
    )

    job = DasymetricRedistributor(config)
    job.run()
    job.verify()

Example
-----
Suppose we have a weighting layer (`covariates/weight_raster.tif`) representing the probability of population residing at particular location. At a certain administrative level (defined by `data/adm_boundary.shp`), the total population is known either from census or demographic projection (summarised in `data/pop_count.csv`). We can dasymetrically redistribute the population using the weighting layer.

The following command can be used:

    python dasymetric.py \\
    covariates/weight_raster.tif \\
    data/adm_boundary.shp \\
    data/pop_count.csv \\
    ID_FIELD \\
    POP_FIELD \\
    out/pop_disaggregated.tif \\
    --workers 4 \\
    --block-size 512

Sometimes, we want to constrain the redistribution over the area where settlement exists. For this case, `mask_path` argument specifying the path to masking layer can be supplied to `DasymetricConfig()`. Alternatively, `--path-mask` argument is available in the command line version.

In case we already have a mastergrid, which is a raster representation of the administrative units with population counts, we can set `geom_path = path_to_mastergrid.tif` to avoid redoing rasterisation of administrative boundaries.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use pyDasymetric in your research, please cite:

```bibtex
@software{pyDasymetric,
	title        = {pyDasymetric: Python package to perform dasymetric redistribution of populations.},
	author       = {Priyatikanto R., Bondarenko M., Nosatiuk B..},
	year         = 2026,
	month        = 6,
	publisher    = {GitHub},
    doi          = {10.5281/zenodo.21815905},
	url          = {https://github.com/wpgp/pyDasymetric},
	version      = {0.0.1}
}
```

## Acknowledgments

- Developed by WorldPop SDI [sdi.worldpop.org](https://sdi.worldpop.org)
