"""Plot distributions of data"""
# import os
import itertools
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmcrameri
import dask.array as da
import dask
import h5py
from datetime import datetime
import re
import pyart

import logging
import logging.config

import plot_utils
import date_utils
import filter


PAIRWISE_QTY_LIST = [
    "DBZH",
    "ZDR",
    "RHOHV",
    "KDP",
    "VRAD",
    "HCLASS",
    "LOG",
    "PMI",
    "CSP",
]
PAIRWISE_QTYS = list(itertools.combinations(PAIRWISE_QTY_LIST, 2))

# Set up logging
with open("logconf.yaml", "rt") as f:
    log_config = yaml.safe_load(f.read())
    f.close()
logging.config.dictConfig(log_config)
logging.captureWarnings(True)
logger = logging.getLogger(Path(__file__).stem)


@dask.delayed
def worker(startdate, enddate, inpath, fileformat, mask, filters=None, n_bins=100):
    # Load data
    times = pd.date_range(startdate, enddate, freq="5T", inclusive="both")

    filenames = {
        t: Path(t.strftime(str(inpath))) / t.strftime(str(fileformat)) for t in times
    }
    filenames = {t: f for t, f in filenames.items() if f.exists()}

    hists = {}
    counts = {}
    bins = {}

    for time, file in filenames.items():
        radar = filter.read_and_filter_radar(file, filters)

        missing_variables = np.setdiff1d(
            [plot_utils.PYART_FIELDS[q] for q in PAIRWISE_QTY_LIST],
            list(radar.fields.keys()),
        )
        if len(missing_variables) > 0:
            logger.warning(
                f"Missing variables {missing_variables} in {file}. Skipping."
            )
            continue

        for fieldname in radar.fields.keys():

            odim_qty = [
                k
                for k in plot_utils.PYART_FIELDS.keys()
                if plot_utils.PYART_FIELDS[k] == fieldname
            ][0]

            field = radar.fields[fieldname]["data"]

            h_, bins_ = np.histogram(
                field[np.where(mask == 1)].ravel(),
                bins=n_bins,
                range=plot_utils.QTY_RANGES[odim_qty],
            )

            if fieldname not in hists.keys():
                hists[odim_qty] = h_
                counts[odim_qty] = sum(h_)
                bins[odim_qty] = bins_
            else:
                hists[odim_qty] += h_
                counts[odim_qty] += sum(h_)

        # Pairwise quantities
        for qty1, qty2 in PAIRWISE_QTYS:
            qty1_field = radar.fields[plot_utils.PYART_FIELDS[qty1]]["data"]
            qty2_field = radar.fields[plot_utils.PYART_FIELDS[qty2]]["data"]

            qty1_field = qty1_field[np.where(mask == 1)]
            qty2_field = qty2_field[np.where(mask == 1)]

            h_, xedges, yedges = np.histogram2d(
                qty1_field.ravel(),
                qty2_field.ravel(),
                bins=n_bins,
                range=[
                    plot_utils.QTY_RANGES[qty1],
                    plot_utils.QTY_RANGES[qty2],
                ],
            )

            qty_pair = f"{qty1}_{qty2}"

            if qty_pair not in hists.keys():
                hists[qty_pair] = h_
                counts[qty_pair] = sum(h_)
                bins[qty_pair] = (xedges, yedges)
            else:
                hists[qty_pair] += h_
                counts[qty_pair] += sum(h_)

    return hists, counts, bins


def main(startdate, enddate, conf, main_outpath, filters=None, nworkers=1):

    args = []
    for radar in conf["radar"]["radars"]:
        for task in conf["radar"]["tasks"]:
            inpath = Path(conf["radar"]["path"].format(radar=radar))
            fileformat = Path(
                conf["radar"]["fileformat"].format(radar=radar, task=task)
            )

            outpath = Path(main_outpath) / radar
            outpath.mkdir(parents=True, exist_ok=True)

            mask = None
            # Load mask from h5 file
            mask_file = Path(conf["masks"][radar]["path"])
            mask_group = conf["masks"][radar]["groups"][task]

            try:
                with h5py.File(mask_file, "r") as f:
                    mask = f[f"{mask_group}/data"][...]
            except (KeyError, OSError, FileNotFoundError) as e:
                raise ValueError(f"Could not load mask from {mask_file}") from e

            args.append((inpath, fileformat, outpath, mask))

    scheduler = "multiprocessing" if nworkers > 1 else "single-threaded"
    # Run in parallel
    for (inpath, fileformat, outpath, mask) in args:

        date_ranges = date_utils.get_chunked_date_range(
            startdate,
            enddate,
            nworkers,
            max_chunk_len=24,
            min_chunk_len=1,
        )

        res = []
        for dr in date_ranges:
            res.append(
                worker(
                    dr[0], dr[1], inpath, fileformat, mask, filters=filters, n_bins=100
                )
            )

        results = dask.compute(
            *res,
            num_workers=nworkers,
            scheduler=scheduler,
            chunksize=1,
        )

        # Combine results
        hists = {}
        counts = {}
        bins = {}
        for r in results:
            for qty in r[0].keys():
                if qty not in hists.keys():
                    hists[qty] = r[0][qty]
                    counts[qty] = r[1][qty]
                    bins[qty] = r[2][qty]
                else:
                    hists[qty] += r[0][qty]
                    counts[qty] += r[1][qty]
        # Save histograms to file
        for qty in hists.keys():
            np.save(
                outpath / f"{qty}_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}_hist.npy",
                hists[qty],
            )
            np.save(
                outpath / f"{qty}_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}_bins.npy",
                bins[qty],
            )
            np.save(
                outpath
                / f"{qty}_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}_counts.npy",
                counts[qty],
            )

        # Plot single variable histograms
        single_var_keys = [k for k in hists.keys() if "_" not in k]
        pair_var_keys = [k for k in hists.keys() if "_" in k]
        ncols = 3
        nrows = np.ceil(len(single_var_keys) / ncols).astype(int)
        fig, axes = plt.subplots(
            figsize=(3.5 * ncols, nrows * 2.1),
            nrows=nrows,
            ncols=ncols,
            squeeze=True,
            sharex=False,
            sharey=False,
        )

        num_plotted_axes = len(single_var_keys)

        for i, fieldname in enumerate(single_var_keys):
            ax = axes.flatten()[i]
            width = bins[fieldname][-1] - bins[fieldname][-2]
            ax.bar(
                bins[fieldname][:-1],
                hists[fieldname],
                width=width,
                align="edge",
                color="k",
                edgecolor="k",
                zorder=10,
            )
            ax.set_title(plot_utils.TITLES[fieldname])
            ax.set_xlabel(plot_utils.COLORBAR_TITLES[fieldname])
            ax.set_xlim(plot_utils.QTY_RANGES[fieldname])

            if fieldname == "HCLASS":
                ax.set_xticklabels(plot_utils.get_HCLASS_labels())

            if fieldname == "PMI":
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        for ax in axes.flat[:num_plotted_axes]:

            # ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            # ax.xaxis.set_minor_formatter(ticker.NullFormatter())


            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.set_ylim(bottom=0)
            ax.grid(which="major", lw=0.5, color="tab:gray", ls="-", zorder=0)
            ax.grid(
                which="minor", lw=0.5, color="tab:gray", ls="-", alpha=0.1, zorder=0
            )

            ax.set_yscale("log")
            ax.set_ylabel("Count")

        for ax in axes.flat[num_plotted_axes:]:
            ax.axis("off")

        outfile = (
            outpath / f"histograms_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}.png"
        )
        outfile.parents[0].mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=600, bbox_inches="tight")

        # Plot pair variable histograms
        ncols = len(PAIRWISE_QTY_LIST) - 1
        # nrows = np.ceil(len(pair_var_keys) / ncols).astype(int)
        nrows = len(PAIRWISE_QTY_LIST) - 1
        fig, axes = plt.subplots(
            # figsize=(4.5 * ncols + 3.0, nrows * 3.5),
            figsize=(5.5 * ncols, 5.0 * nrows),
            nrows=nrows,
            ncols=ncols,
            squeeze=True,
            sharex=False,
            sharey=False,
            constrained_layout=True,
        )

        num_plotted_axes = len(pair_var_keys)

        hist_cmap = "cmc.imola"
        bounds = np.logspace(1, 4, 40)
        num_c = len(bounds)
        hist_cmap = plt.get_cmap(hist_cmap, num_c)
        hist_norm = mpl.colors.BoundaryNorm(bounds, hist_cmap.N)

        cbar_ax_kws = {
            "width": "3%",  # width = 5% of parent_bbox width
            "height": "100%",
            "loc": "lower left",
            "bbox_to_anchor": (1.01, 0.0, 1, 1),
            "borderpad": 0,
        }

        for i, fieldname in enumerate(pair_var_keys):
            qty1, qty2 = fieldname.split("_")
            col = PAIRWISE_QTY_LIST.index(qty1)
            row = PAIRWISE_QTY_LIST.index(qty2) - 1
            ax = axes[row, col]

            bins1 = bins[fieldname][0]
            bins2 = bins[fieldname][1]
            hist = hists[fieldname]
            hist = np.ma.masked_where(hist < 10, hist)

            xx, yy = np.meshgrid(bins1, bins2)
            xwidth = np.diff(bins1)[0]
            ywidth = np.diff(bins2)[0]
            ax.pcolormesh(
                xx + xwidth / 2,
                yy + ywidth / 2,
                hist.T,
                cmap=hist_cmap,
                norm=hist_norm,
                shading="auto",
                rasterized=True,
            )
            # cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)
            # cbar = plt.colorbar(
            #     mpl.cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap),
            #     cax=cax,
            #     extend="max",
            #     format=mpl.ticker.LogFormatterSciNotation(labelOnlyBase=True),
            #     ticks=mpl.ticker.LogLocator(),
            # )
            # cbar.set_label("Frequency", labelpad=-1)

            ax.set_title(f"{plot_utils.TITLES[qty1]} / {plot_utils.TITLES[qty2]}")
            ax.set_xlabel(plot_utils.COLORBAR_TITLES[qty1])
            ax.set_ylabel(plot_utils.COLORBAR_TITLES[qty2])
            ax.set_xlim(plot_utils.QTY_RANGES[qty1])
            ax.set_ylim(plot_utils.QTY_RANGES[qty2])

            if qty1 == "HCLASS":
                ax.set_xticklabels(plot_utils.get_HCLASS_labels())
            if qty2 == "HCLASS":
                ax.set_yticklabels(plot_utils.get_HCLASS_labels())

            if qty1 == "PMI":
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            if qty2 == "PMI":
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

            ax.grid(which="major", lw=0.5, color="tab:gray", ls="-", zorder=0)
            ax.grid(
                which="minor", lw=0.5, color="tab:gray", ls="-", alpha=0.5, zorder=0
            )

        for i in range(nrows):
            for j in range(ncols):
                if i < j:
                    axes[i, j].axis("off")

        # fig.subplots_adjust(wspace=1, hspace=0.1)
        fig.subplots_adjust(wspace=2.0, hspace=0.4)
        outfile = (
            outpath
            / f"histograms_pairs_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}.png"
        )
        outfile.parents[0].mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=600)  # , bbox_inches="tight")

        # Plot colorbar separately
        fig, ax = plt.subplots(figsize=(0.7, 4.5))
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap),
            cax=ax,
            extend="max",
            format=mpl.ticker.LogFormatterSciNotation(labelOnlyBase=True),
            ticks=mpl.ticker.LogLocator(),
        )
        cbar.set_label("Frequency", labelpad=-1)
        outfile = (
            outpath
            / f"histograms_pairs_colorbar_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}.png"
        )
        outfile.parents[0].mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("startdate", type=str, help="Start date")
    argparser.add_argument("enddate", type=str, help="End date")
    argparser.add_argument("confpath", type=str, help="Configuration file path")
    argparser.add_argument("outpath", type=str, help="Output file path")
    argparser.add_argument(
        "-n", "--nworkers", default=1, type=int, help="Number of workers"
    )
    argparser.add_argument(
        "-f", "--filter", type=str, default=None, help="Filter configuration file"
    )
    args = argparser.parse_args()

    # Load configuration from yaml file
    with open(args.confpath, "r") as f:
        conf = yaml.safe_load(f)

    outpath = Path(args.outpath)

    startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
    enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")

    # Read filter config
    if args.filter is not None:
        filter_funcs = filter.create_filters_from_config(args.filter)
    else:
        filter_funcs = None

    plt.style.use("distributions.mplstyle")
    main(
        startdate, enddate, conf, outpath, filters=filter_funcs, nworkers=args.nworkers
    )
