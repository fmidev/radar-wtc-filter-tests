"""Plot comparison of filtered and unfiltered PPIs."""
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

import filter
import utils


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("inpath", type=str, help="File input path")
    argparser.add_argument("filterconfig", type=str, help="Path to filter config file")
    argparser.add_argument("outpath", type=str, help="File output path")
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
    argparser.add_argument(
        "-q",
        "--qtys",
        type=str,
        nargs="+",
        help="Quantity to plot",
        default=["DBZH", "ZDR", "VRAD", "HCLASS", "PMI", "LOG"],
    )
    argparser.add_argument("-a", "--alpha", type=float, help="Mask alpha", default=0.1)
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

    filter_funcs = filter.create_filters_from_config(args.filterconfig)

    qtys = args.qtys

    outpath = Path(args.outpath)
    outpath.mkdir(exist_ok=True, parents=True)

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

        # Fin

        fradar = filter.read_and_filter_radar(fn, filters=filter_funcs, mask_field=mask)
        radar = pyart.io.read(fn)
        if mask is not None:
            mask_field = fradar.fields["mask"].copy()
            radar.add_field("mask", mask_field)

        # Create the figure
        utils.plot_ppi_comparison(
            radar,
            fradar,
            qtys,
            max_dist=250,
            outdir=outpath,
            markers=None,
            ext="png",
            mask="mask" if mask is not None else None,
            mask_alpha=args.alpha,
        )
        plt.close()
