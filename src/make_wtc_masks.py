"""Make masks that show wind turbine locations for each radar."""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import pyart
import yaml
import wradlib as wrl

pyart.load_config(os.environ.get("PYART_CONFIG"))


def load_turbine_data(csv_path, skip_rows=1):
    """
    Load and filter wind turbine data from CSV file.
    
    Args:
        csv_path: Path to CSV file containing turbine data
        skip_rows: Number of header rows to skip
        
    Returns:
        np.ndarray: Array of shape (n_turbines, 3) with [lon, lat, alt] in meters
        
    Raises:
        ValueError: If no wind turbines found or elevation column missing
    """
    df = pd.read_csv(csv_path, sep=";", skiprows=skip_rows)
    df = df[df["TYPE"] == "Wind turbine"]

    if len(df) == 0:
        raise ValueError("No wind turbines found in file")

    if "ELEV MSL (m)" not in df.columns and "ELEV MSL (FT)" in df.columns:
        # Transform from feet to meters
        df["ELEV MSL (m)"] = (
            df["ELEV MSL (FT)"].apply(pd.to_numeric, errors="coerce") * 0.3048
        )
    else:
        raise ValueError("Elevation column not found")

    turbine_lonlatalt = np.array(
        [
            df["LONG"].apply(pd.to_numeric, errors="coerce").values,
            df["LAT"].apply(pd.to_numeric, errors="coerce").values,
            df["ELEV MSL (m)"].apply(pd.to_numeric, errors="coerce").values,
        ]
    ).T
    # Filter out turbines with missing data
    turbine_lonlatalt = turbine_lonlatalt[~np.isnan(turbine_lonlatalt).any(axis=1)]
    
    return turbine_lonlatalt


def convert_turbines_to_radar_coords(turbine_lonlatalt, radar_lonlatalt):
    """
    Convert turbine geographic coordinates to radar spherical coordinates.
    
    Args:
        turbine_lonlatalt: Array of shape (n_turbines, 3) with [lon, lat, alt]
        radar_lonlatalt: Array of shape (3,) with radar [lon, lat, alt]
        
    Returns:
        tuple: (ranges, azims, elevs) - All as numpy arrays
    """
    turbine_x, turbine_y = pyart.core.geographic_to_cartesian_aeqd(
        turbine_lonlatalt[:, 0], turbine_lonlatalt[:, 1], *radar_lonlatalt[:2]
    )
    turbine_xyz = np.array([turbine_x, turbine_y, turbine_lonlatalt[:, 2]]).T

    ranges, azims, elevs = wrl.georef.xyz_to_spherical(
        turbine_xyz, altitude=radar_lonlatalt[2]
    )
    # Round azimuths to data resolution
    azims = np.rint(np.floor(azims)).astype(int)
    
    return ranges, azims, elevs


def get_elevation_buffer(radar_elevs, buffer_config):
    """
    Get elevation buffer value based on radar elevation.
    
    Args:
        radar_elevs: Array of radar elevations at turbine azimuths
        buffer_config: List of dicts with 'elev_interval' and 'buffer' keys
        
    Returns:
        float: Buffer value in degrees
    """
    buffer = 0.5  # Default
    mean_elev = np.mean(radar_elevs)
    
    for d in buffer_config:
        if d["elev_interval"][0] <= mean_elev <= d["elev_interval"][1]:
            buffer = d["buffer"]
    
    return buffer


def filter_turbines_by_elevation(elevs, radar_elevs, elev_buffer):
    """
    Filter turbines based on elevation relative to radar beam.
    
    Args:
        elevs: Turbine elevations in degrees
        radar_elevs: Radar beam elevations at turbine azimuths
        elev_buffer: Buffer in degrees
        
    Returns:
        np.ndarray: Boolean mask indicating which turbines to include
    """
    return (elevs > radar_elevs) | (abs(elevs - radar_elevs) <= elev_buffer)


def apply_turbine_mask(mask, turbine_range, turbine_azim, 
                       buffer_range_before, buffer_range_after, buffer_azim):
    """
    Apply mask for a single turbine with range and azimuth buffers.
    
    Handles azimuth wraparound at 0/360 degree boundary.
    
    Args:
        mask: Boolean mask array to modify (in-place)
        turbine_range: Turbine range bin index
        turbine_azim: Turbine azimuth bin index
        buffer_range_before: Range buffer before turbine (bins)
        buffer_range_after: Range buffer after turbine (bins)
        buffer_azim: Azimuth buffer (bins)
    """
    r_min = max(0, turbine_range - buffer_range_before)
    r_max = min(turbine_range + buffer_range_after, mask.shape[1])
    a_min = turbine_azim - buffer_azim
    a_max = turbine_azim + buffer_azim + 1
    
    # Handle azimuth wraparound at 360/0 degrees
    if a_min < 0 or a_max > mask.shape[0]:
        # Wrap around case
        if a_min < 0:
            mask[a_min:, r_min:r_max] = True
            mask[:a_max, r_min:r_max] = True
        else:
            mask[a_min:, r_min:r_max] = True
            mask[:a_max % mask.shape[0], r_min:r_max] = True
    else:
        mask[a_min:a_max, r_min:r_max] = True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration file path")
    argparser.add_argument("--num-skip-rows", type=int, default=1, help="Number of rows to skip in the turbine file")
    args = argparser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Read radar objects
    radar_objects = {}
    for radar, files in config["example_radar_files"].items():
        radar_objects[radar] = {}
        for file in files:
            for dataset in config["process_datasets"]:
                try:
                    radar_objects[radar][(file, dataset)] = pyart.aux_io.read_odim_h5(
                        file, include_datasets=[dataset]
                    )
                except IndexError:
                    print(f"Dataset {dataset} not found in file {file}, skipping"),

    datasets = config["process_datasets"]

    # Load turbine locations
    turbine_lonlatalt = load_turbine_data(
        config["wind_turbine_list"], 
        skip_rows=args.num_skip_rows
    )

    for radar, data in radar_objects.items():
        first_key = list(data.keys())[0]
        # Get radar coordinates
        radar_lonlatalt = np.array(
            [
                data[first_key].longitude["data"].item(),
                data[first_key].latitude["data"].item(),
                data[first_key].altitude["data"].item(),
            ]
        )

        # Convert turbines to radar spherical coordinates
        ranges, azims, elevs = convert_turbines_to_radar_coords(
            turbine_lonlatalt, radar_lonlatalt
        )

        # Initialize mask file
        mask_path = Path(config["output_path"]) / config["output_filename"].format(radar_name=radar.lower())
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy metadata from first radar file
        with h5py.File(mask_path, "w") as f_dest, h5py.File(first_key[0], "r") as f_src:
            # Copy metadata from radar file
            f_src.copy(f_src["where"], f_dest, "where")
            f_src.copy(f_src["what"], f_dest, "what")
            f_src.copy(f_src["how"], f_dest, "how")

        for (infile, dataset), radar_obj in data.items():
            # Round ranges to data resolution
            ranges_ = np.rint(ranges / radar_obj.range["meters_between_gates"]).astype(
                int
            )
            # Remove turbines outside radar range
            selected_turbines = ranges_ < radar_obj.range["data"].shape[0]
            ranges_ = ranges_[selected_turbines]
            azims_ = azims[selected_turbines]
            elevs_ = elevs[selected_turbines]

            # Create mask for each wind turbine
            mask = np.zeros(
                radar_obj.fields["reflectivity_horizontal"]["data"].shape, dtype=bool
            )
            radar_elevs = radar_obj.elevation["data"][azims_]

            # Get elevation buffer and filter turbines
            elev_buffer = get_elevation_buffer(radar_elevs, config["elev_buffer_degrees"])
            selected_turbines_elev = filter_turbines_by_elevation(
                elevs_, radar_elevs, elev_buffer
            )

            # Calculate buffer sizes
            buffer_range_before = int(
                np.ceil(
                    config["range_buffer_before_meters"]
                    / radar_obj.range["meters_between_gates"]
                )
            )
            buffer_range_after = int(
                np.ceil(
                    config["range_buffer_after_meters"]
                    / radar_obj.range["meters_between_gates"]
                )
            )
            buffer_azim = int(np.ceil(config["azim_buffer_degrees"] / 1.0))

            # Create mask for each wind turbine
            for i, (r, a, e) in enumerate(zip(ranges_, azims_, elevs_)):
                if selected_turbines_elev[i]:
                    apply_turbine_mask(
                        mask, r, a,
                        buffer_range_before, buffer_range_after, buffer_azim
                    )

            # Save mask
            with h5py.File(mask_path, "a") as f:
                # Get output dataset name as the highest dataset number + 1
                dataset_no = (
                    max(
                        [
                            int(re.findall(r"\d+", d)[0])
                            for d in f.keys()
                            if "dataset" in d
                        ]
                        + [0]
                    )
                    + 1
                )
                dsname = f"dataset{dataset_no}"
                dset = f.require_dataset(
                    f"{dsname}/data1/data",
                    shape=mask.shape,
                    data=mask,
                    dtype=bool,
                    compression="gzip",
                    compression_opts=9,
                )
                dset.attrs["CLASS"] = np.bytes_("IMAGE")
                dset.attrs["IMAGE_VERSION"] = np.bytes_("1.2")
                how = f[f"{dsname}/data1"].require_group("how")
                what = f[f"{dsname}/data1"].require_group("what")
                what.attrs["gain"] = 1.0
                what.attrs["offset"] = 0.0
                what.attrs["nodata"] = 0.0
                what.attrs["quantity"] = "mask"
                what.attrs["undetect"] = 0.0

                with h5py.File(infile, "r") as f_src:
                    f_src.copy(f_src[f"/{dataset}/where"], f[f"/{dsname}"], "where")
                    f_src.copy(f_src[f"/{dataset}/what"], f[f"/{dsname}"], "what")
                    f_src.copy(f_src[f"/{dataset}/how"], f[f"/{dsname}"], "how")
