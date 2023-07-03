"""Plot distributions of data"""
# import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


@dask.delayed
def worker(startdate, enddate, inpath, fileformat, mask, n_bins=100):
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
        radar = pyart.io.read(file)

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

    return hists, counts, bins


def main(startdate, enddate, conf, outpath, nworkers=1):

    args = []
    for radar in conf["radar"]["radars"]:
        for task in conf["radar"]["tasks"]:
            inpath = Path(conf["radar"]["path"].format(radar=radar))
            fileformat = Path(
                conf["radar"]["fileformat"].format(radar=radar, task=task)
            )

            outpath = Path(outpath) / radar
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
            res.append(worker(dr[0], dr[1], inpath, fileformat, mask, n_bins=100))

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

        ncols = 3
        nrows = np.ceil(len(hists.keys()) / ncols).astype(int)
        fig, axes = plt.subplots(
            figsize=(3.5 * ncols, nrows * 2.1),
            nrows=nrows,
            ncols=ncols,
            squeeze=True,
        )

        for i, fieldname in enumerate(hists.keys()):
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

        for ax in axes.flat:

            # ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            # ax.xaxis.set_minor_formatter(ticker.NullFormatter())

            # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))s
            ax.set_ylim(bottom=0)
            ax.grid(which="major", lw=0.5, color="tab:gray", ls="-", zorder=0)
            ax.grid(
                which="minor", lw=0.5, color="tab:gray", ls="-", alpha=0.1, zorder=0
            )

            # ax.set_yscale("log")

            ax.set_ylabel("Count")

        outfile = (
            outpath / f"histograms_{startdate:%Y%m%d%H%M}_{enddate:%Y%m%d%H%M}.png"
        )
        outfile.parents[0].mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=600, bbox_inches="tight")

    # files = {
    #     datetime.strptime(p.name.split("_")[0], filename_pattern.split("_")[0]): p
    #     for p in inpath.glob(filename_glob)
    # }
    # files = dict(sorted(files.items()))

    # if mask is None:
    #     mask = np.ones((360, 500))

    # n_bins = 100
    # batch_size = 15

    # hists = {}
    # counts = {}
    # bins = {}

    # datasets = []
    # for time, file in files.items():
    #     radar = pyart.io.read(file)

    #     data = {
    #         fieldname: (
    #             ("time", "mask_idx"),
    #             radar.fields[fieldname]["data"][np.where(mask == 1)].ravel()[
    #                 np.newaxis, ...
    #             ],
    #         )
    #         for fieldname in radar.fields.keys()
    #     }
    #     ds = xr.Dataset(
    #         data_vars=data,
    #         coords={"time": [time], "mask_idx": np.arange(0, np.sum(mask == 1))},
    #     )
    #     datasets.append(ds)

    #     for fieldname in radar.fields.keys():

    #         odim_qty = [
    #             k
    #             for k in plot_utils.PYART_FIELDS.keys()
    #             if plot_utils.PYART_FIELDS[k] == fieldname
    #         ][0]

    #         field = radar.fields[fieldname]["data"]

    #         h_, bins_ = np.histogram(
    #             field[np.where(mask == 1)].ravel(),
    #             bins=n_bins,
    #             range=plot_utils.QTY_RANGES[odim_qty],
    #         )

    #         if fieldname not in hists.keys():
    #             hists[odim_qty] = h_
    #             counts[odim_qty] = sum(h_)
    #             bins[odim_qty] = bins_
    #         else:
    #             hists[odim_qty] += h_
    #             counts[odim_qty] += sum(h_)

    # # Plot histogram
    # data_ds = xr.concat(datasets, dim="time")
    # df = data_ds.to_dataframe()
    # df["time"] = df.index.get_level_values(0)
    # df["hour"] = df["time"].dt.strftime("%Y-%m-%d %H")

    # for label, group in df.groupby("hour"):
    #     # Plot pairwise scatter plots colored by hour
    #     g = sns.pairplot(
    #         group,
    #         # hue="hour",
    #         vars=[
    #             "reflectivity",
    #             "velocity",
    #             "spectrum_width",
    #             "signal_to_noise_ratio",
    #             "polarimetric_meteo_index",
    #             "radar_echo_classification",
    #             "log_signal_to_noise_ratio",
    #             "clutter_power_ratio",
    #         ],
    #         # plot_kws={"s": 1, "alpha": 0.3},
    #         kind="hist",
    #         palette="cubehelix",
    #         dropna=True,
    #         # corner=True,
    #     )
    #     plt.savefig(f"pairplot-{label}.png", dpi=600, bbox_inches="tight")
    #     plt.close()
    # diag_kws={"bins": 100},

    # import ipdb

    # ipdb.set_trace()

    # ncols = 3
    # nrows = np.ceil(len(hists.keys()) / ncols).astype(int)
    # fig, axes = plt.subplots(
    #     figsize=(3.5 * ncols, nrows * 2.1),
    #     nrows=nrows,
    #     ncols=ncols,
    #     squeeze=True,
    # )

    # for i, fieldname in enumerate(hists.keys()):
    #     ax = axes.flatten()[i]
    #     width = bins[fieldname][-1] - bins[fieldname][-2]
    #     ax.bar(
    #         bins[fieldname][:-1],
    #         hists[fieldname],
    #         width=width,
    #         align="edge",
    #         color="k",
    #         edgecolor="k",
    #         zorder=10,
    #     )
    #     ax.set_title(plot_utils.TITLES[fieldname])
    #     ax.set_xlabel(plot_utils.COLORBAR_TITLES[fieldname])
    #     ax.set_xlim(plot_utils.QTY_RANGES[fieldname])

    # for ax in axes.flat:

    #     # ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #     # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    #     # ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #     # ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    #     # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))s
    #     ax.set_ylim(bottom=0)
    #     ax.grid(which="major", lw=0.5, color="tab:gray", ls="-", zorder=0)
    #     ax.grid(which="minor", lw=0.5, color="tab:gray", ls="-", alpha=0.1, zorder=0)

    #     # ax.set_yscale("log")

    #     ax.set_ylabel("Count")

    # outpath = Path(args.outpath)
    # outpath.parents[0].mkdir(parents=True, exist_ok=True)
    # fig.savefig(outpath, dpi=600, bbox_inches="tight")


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
    args = argparser.parse_args()

    # Load configuration from yaml file
    with open(args.confpath, "r") as f:
        conf = yaml.safe_load(f)

    outpath = Path(args.outpath)

    startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
    enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")

    # Set up logging
    with open("logconf.yaml", "rt") as f:
        log_config = yaml.safe_load(f.read())
        f.close()
    logging.config.dictConfig(log_config)
    logging.captureWarnings(True)
    logger = logging.getLogger(Path(__file__).stem)

    plt.style.use("distributions.mplstyle")
    main(startdate, enddate, conf, outpath, nworkers=args.nworkers)
