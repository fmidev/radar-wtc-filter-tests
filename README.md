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
    # If only_inside_mask is true, only data inside the mask will be used
    # if no mask is given, this is ignored
    only_inside_mask: true
    mask_field_name: "mask"
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

usage: plot_ppis.py [-h] [--no-filter] [--filepattern FILEPATTERN] [-m MASK] [--mask-group MASK_GROUP] inpath filterconfig outpath

positional arguments:
  inpath                File input path
  filterconfig          Path to filter config file
  outpath               File output path

options:
  -h, --help            show this help message and exit
  --no-filter           Do not apply filters
  --filepattern FILEPATTERN, -f FILEPATTERN
                        Filename pattern
  -m MASK, --mask MASK  Mask file
  --mask-group MASK_GROUP
                        Mask group

```

The `<filepattern>` argument is a regex expression that is used to match the files in the input folder. The default is `%Y%m%d%H%M_*.PPI1_A.raw` which matches all PPI1_A Sigmet raw files. Any files that Py-ART can read can be used.

A mask for filtering can be specified with the `mask` argument. The value should be a HDF5 file created with with the `make_wtc_mask.py` script (or similar file). The `mask-group` argument specifies the group in the HDF5 file to use as the mask.

Note that the Py-ART config file needs to be updated to read some extra fields in the Sigmet files that are not available by default. An example config file is given at `~/.pyart_config.py`.

### Apply filter

```bash
python plot_ppis.py <input-folder> <filter-config.yaml> <output-path>
```

### No filter applied

```bash
python plot_ppis.py <input-folder> <filter-config.yaml> <output-path> --no-filter
```

## Plotting a comparison of filtered and original data

```bash
> python plot_ppi_comparison.py --help

usage: plot_ppi_comparison.py [-h] [--filepattern FILEPATTERN] [-m MASK] [--mask-group MASK_GROUP] [-q QTYS [QTYS ...]] inpath filterconfig outpath

Plot comparison of filtered and unfiltered PPIs.

positional arguments:
  inpath                File input path
  filterconfig          Path to filter config file
  outpath               File output path

options:
  -h, --help            show this help message and exit
  --filepattern FILEPATTERN, -f FILEPATTERN
                        Filename pattern
  -m MASK, --mask MASK  Mask file
  --mask-group MASK_GROUP
                        Mask group
  -q QTYS [QTYS ...], --qtys QTYS [QTYS ...]
                        Quantity to plot
```

The arguments are similar to the `plot_ppis.py` script.

## Creating wind turbine masks

```bash
> python make_wtc_masks.py --help

usage: make_wtc_masks.py [-h] config

Make masks that show wind turbine locations for each radar.

positional arguments:
  config      Configuration file path

options:
  -h, --help  show this help message and exit

```

The configuration file is a YAML file that specifies the mask configuration and template radar files. An example is given below:

```yaml
example_radar_files:
  ANJ: example_radar_files/202307010000_fianj_PVOL.h5
  KAN: example_radar_files/202307010000_fikan_PVOL.h5
  KES: example_radar_files/202307010000_fikes_PVOL.h5
  KOR: example_radar_files/202307010000_fikor_PVOL.h5
  KUO: example_radar_files/202307010000_fikuo_PVOL.h5
  LUO: example_radar_files/202307010000_filuo_PVOL.h5
  NUR: example_radar_files/202307010000_finur_PVOL.h5
  PET: example_radar_files/202307010000_fipet_PVOL.h5
  UTA: example_radar_files/202307010000_fiuta_PVOL.h5
  VIH: example_radar_files/202307010000_fivih_PVOL.h5
  VIM: example_radar_files/202307010000_fivim_PVOL.h5

# Lentoeste file
wind_turbine_list: ef_efin_area1_obstdata_26_jan_2023.csv

process_datasets:
  - dataset1
  - dataset2

# Buffer to use around wind turbine
elev_buffer_degrees: 0.6
range_buffer_before_meters: 1000
range_buffer_after_meters: 10000
azim_buffer_degrees: 2.0

output_path: turbine_masks
output_filene: fi{radar_name}_turbine_mask.h5
```

The `wind_turbine_list` file is a CSV file with at least the following columns: `TYPE`, `LONG`, `LAT`, `ELEV MSL (m)`

## Calculating radar variable distributions

```bash
> python calculate_distributions.py -h

usage: calculate_distributions.py [-h] [-n NWORKERS] [-f FILTER] startdate enddate confpath outpath

Plot distributions of data

positional arguments:
  startdate             Start date
  enddate               End date
  confpath              Configuration file path
  outpath               Output file path

options:
  -h, --help            show this help message and exit
  -n NWORKERS, --nworkers NWORKERS
                        Number of workers
  -f FILTER, --filter FILTER
                        Filter configuration file

```

The configuration file is a YAML file that specifies the radar data to use and the masks that used to select data. An example is given below:

```yaml
radar:
  path: "/arch/radar/raw/%Y/%m/%d/iris/raw/{radar}/"
  fileformat: "%Y%m%d%H%M_{radar}.{task}.raw"
  tasks: [PPI1_A]
  radars: [UTA, KAN, VIM]

masks:
  VIM:
    path: masks/202303130030_fivim_PVOL_mask.h5
    groups:
      PPI1_A: dataset1/data1
  KAN:
    path: masks/202306060145_fikan_PVOL_mask.h5
    groups:
      PPI1_A: dataset1/data1
  UTA:
    path: masks/202306060315_fiuta_PVOL_mask.h5
    groups:
      PPI1_A: dataset1/data1
```
