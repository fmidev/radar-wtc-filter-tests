import argparse
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pyart
import utils
import yaml
from scipy.ndimage import generic_filter

pyart.load_config(os.environ.get("PYART_CONFIG"))


def data_filter(radar, config):
    primary_mask = np.ones_like(radar.fields["reflectivity"]["data"], dtype=bool)
    for field, conf in config["primary_filter"].items():
        if field in radar.fields.keys():
            if "upper" in conf.keys() and conf["upper"] is not None:
                primary_mask = np.logical_and(
                    primary_mask,
                    radar.fields[field]["data"] < conf["upper"],
                )
            if "lower" in conf.keys() and conf["lower"] is not None:
                primary_mask = np.logical_and(
                    primary_mask,
                    radar.fields[field]["data"] > conf["lower"],
                )
            if "in" in conf.keys() and conf["in"] is not None:
                primary_mask = np.logical_and(
                    primary_mask,
                    np.isin(radar.fields[field]["data"], conf["in"]),
                )

    # Apply secondary in a window around pixels removed by primary filter
    if "secondary_filter" in config.keys():
        secondary_mask = np.ones_like(radar.fields["reflectivity"]["data"], dtype=bool)
        for field, conf in config["secondary_filter"]["data"].items():
            if field in radar.fields.keys():
                if "upper" in conf.keys() and conf["upper"] is not None:
                    secondary_mask = np.logical_and(
                        secondary_mask,
                        radar.fields[field]["data"] < conf["upper"],
                    )
                if "lower" in conf.keys() and conf["lower"] is not None:
                    secondary_mask = np.logical_and(
                        secondary_mask,
                        radar.fields[field]["data"] > conf["lower"],
                    )
                if "in" in conf.keys() and conf["in"] is not None:
                    secondary_mask = np.logical_and(
                        secondary_mask,
                        np.isin(radar.fields[field]["data"], conf["in"]),
                    )

        # Apply secondary filter in a window around primary filter
        sum_primary = generic_filter(
            primary_mask.astype(int),
            np.sum,
            size=config["secondary_filter"]["window_size"],
        )
        mask = primary_mask | (secondary_mask & (sum_primary > 1))
    else:
        mask = primary_mask

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_gates(mask)
    radar.add_filter(
        gatefilter,
        replace_existing=True,
    )
    return radar


def read_and_filter_radar(filename, filters=None, mask_field=None):
    """Read a radar file and apply filters to it."""
    radar = pyart.io.read(filename)

    if mask_field is not None:
        radar.add_field_like(
            "cross_correlation_ratio",
            "mask",
            mask_field.copy(),
            replace_existing=False,
        )

    if filters is not None:
        for filter in filters:
            if len(filter) > 1:
                filter_kwargs = filter[1]
                filter = filter[0]
            else:
                filter_kwargs = {}
            radar = filter(radar, filter_kwargs)
    return radar


def calculate_accumulation(
    radars,
    median_fields=[
        "radar_echo_classification",
    ],
):
    accum_obj = deepcopy(radars[0])
    fields = accum_obj.fields.keys()
    n_objs = len(radars)

    for field in fields:
        accum_obj.fields[field]["data"] = np.ma.zeros_like(
            accum_obj.fields[field]["data"]
        )

        for radar in radars:
            try:
                accum_obj.fields[field]["data"] += radar.fields[field]["data"]
            except KeyError:
                print(f"key {field} not found in radar {radar.time['units']}")

        # if field in median_fields:
        #     accum_obj.fields[field]["data"] = np.ma.median(
        accum_obj.fields[field]["data"] /= n_objs

    return accum_obj


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("inpath", type=str, help="File input path")
    argparser.add_argument("filterconfig", type=str, help="Path to filter config file")
    argparser.add_argument("outpath", type=str, help="File output path")
    argparser.add_argument(
        "--no-filter", action="store_true", help="Do not apply filters"
    )
    argparser.add_argument(
        "--filepattern",
        "-f",
        type=str,
        default="%Y%m%d%H%M_*.PPI1_A.raw",
        help="Filename pattern",
    )
    argparser.add_argument("-m", "--mask", type=str, help="Mask file")
    argparser.add_argument(
        "--mask-group", type=str, default="dataset1/data1", help="Mask group"
    )
    args = argparser.parse_args()

    inpath = Path(args.inpath)

    with open(args.filterconfig) as f:
        config = yaml.safe_load(f)
        filterconfig = config["filters"]

    filename_pattern = args.filepattern
    filename_glob = re.sub(
        "(%[%YmdHMS])+",
        "*",
        filename_pattern,
    )

    files = {
        datetime.strptime(p.name.split("_")[0], filename_pattern.split("_")[0]): p
        for p in inpath.glob(filename_glob)
    }
    files = dict(sorted(files.items()))

    if args.no_filter:
        filters = None
    else:
        filters = [(data_filter, filterconfig[k]) for k in filterconfig.keys()]

    qtys = [
        "DBZH",
        "ZDR",
        "VRAD",
        # "WRAD",
        # "KDP",
        "HCLASS",
        "PMI",
        # "SQI",
        # "CSP",
        "LOG",
    ]

    outpath = Path(args.outpath)
    outpath.mkdir(exist_ok=True, parents=True)

    prefix = "Filtered " if not args.no_filter else "Unfiltered "

    if args.mask is not None:
        with h5py.File(args.mask, "r") as f:
            mask = f[f"{args.mask_group}/data"][...]
            mask = ~mask
            mask = mask.astype(float)
            mask[mask == 1] = np.nan
            mask = np.ma.masked_invalid(mask)
    else:
        mask = None

    for timestamp, fn in files.items():
        radar = read_and_filter_radar(fn, filters=filters, mask_field=mask)

        utils.plot_ppi_fig(
            radar,
            qtys,
            ncols=2,
            max_dist=150,
            outdir=outpath,
            markers=None,
            ext="png",
            title_prefix=prefix,
            mask="mask" if mask is not None else None,
        )
        plt.close()
