# Test code for filtering radar data

## Build Docker container

```bash
docker build --pull --rm -f "Dockerfile" -t wtcfiltering:latest "."
```

You can use the script `run_container.sh` to run the container. Update mounts to suit your setup.

## Specifying filter

The filters to be applied to the data are specified in the `filters.yaml` file. The filters are applied to the data in polar coordinates. The file is structured as follows:

```yaml
filters:
  # Thresholds interpreted as lower < value < upper
  tail_filter: # filter name
    primary_filter: # primary filter applied to whole image
      # filter threshold values for data that is removed
      # Variable name needs to match pyart field name
      velocity:
        upper: 1.5
        lower: -1.5
      log_signal_to_noise_ratio:
        upper: 5.0
        lower: null
      radar_echo_classification:
        in: [1]
        upper: null
        lower: null
  turbine_filter:
    primary_filter: # primary filter applied to whole image
      reflectivity:
        upper: null
        lower: 30
      polarimetric_meteo_index:
        upper: 0.45
        lower: null
      log_signal_to_noise_ratio:
        upper: null
        lower: 30.0
    secondary_filter: # secondary filter applied in a window around pixels removed by primary filter
      # Window size [azimuth, range]
      window_size: [5, 5]
      data:
        log_signal_to_noise_ratio:
          upper: null
          lower: 20.0
```

## Plotting PPI figures

```bash
> python plot_ppi.py --help

usage: plot_ppis.py [-h] [--no-filter] [--filepattern FILEPATTERN] inpath filterconfig outpath

positional arguments:
  inpath                File input path
  filterconfig          Path to filter config file
  outpath               File output path

options:
  -h, --help            show this help message and exit
  --no-filter           Do not apply filters
  --filepattern FILEPATTERN, -f FILEPATTERN
                        Filename pattern
```

The `<filepattern>` argument is a regex expression that is used to match the files in the input folder. The default is `%Y%m%d%H%M_*.PPI1_A.raw` which matches all PPI1_A Sigmet raw files. Any files that Py-ART can read can be used.

Note that the Py-ART config file needs to be updated to read some extra fields in the Sigmet files that are not available by default. An example config file is given at `~/.pyart_config.py`.

### Apply filter

```bash
python plot_ppis.py <input-folder> <filter-config.yaml> <output-path>
```

### No filter applied

```bash
python plot_ppis.py <input-folder> <filter-config.yaml> <output-path> --no-filter
```
