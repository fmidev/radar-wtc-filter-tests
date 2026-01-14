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

    # Read turbine locations
    df = pd.read_csv(config["wind_turbine_list"], sep=";", skiprows=args.num_skip_rows)
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

    for radar, data in radar_objects.items():
        first_key = list(data.keys())[0]
        # Calculate range, azimuth and altitude for each wind turbine
        radar_lonlatalt = np.array(
            [
                data[first_key].longitude["data"].item(),
                data[first_key].latitude["data"].item(),
                data[first_key].altitude["data"].item(),
            ]
        )

        turbine_x, turbine_y = pyart.core.geographic_to_cartesian_aeqd(
            turbine_lonlatalt[:, 0], turbine_lonlatalt[:, 1], *radar_lonlatalt[:2]
        )
        turbine_xyz = np.array([turbine_x, turbine_y, turbine_lonlatalt[:, 2]]).T

        ranges, azims, elevs = wrl.georef.xyz_to_spherical(turbine_xyz, altitude=radar_lonlatalt[2])
        # Round azimuths to data resolution
        azims = np.rint(np.floor(azims)).astype(int)

        # Initialize mask file
        mask_path = Path(config["output_path"]) / config["output_filename"].format(radar_name=radar.lower())

        # if not mask_path.exists():
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

            # Select turbines that are either above the radar elevation or within the buffer
            # Get buffer value from config
            buffer = 0.5
            for d in config["elev_buffer_degrees"]:
                if (
                    d["elev_interval"][0]
                    <= np.mean(radar_elevs)
                    <= d["elev_interval"][1]
                ):
                    buffer = d["buffer"]
            selected_turbines_elev = (elevs_ > radar_elevs) | (
                abs(elevs_ - radar_elevs) <= buffer
            )

            # Calculate buffer size
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
                    # Add buffer to range and azimuth
                    r_min = max(0, r - buffer_range_before)
                    r_max = min(r + buffer_range_after, mask.shape[1])
                    a_min = max(0, a - buffer_azim)
                    a_max = min(a + buffer_azim, mask.shape[0])
                    mask[a_min:a_max, r_min:r_max] = True

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
