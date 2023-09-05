import pyart
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

import plot_utils


def get_radar_coords(lon, lat, radar):
    return pyart.core.geographic_to_cartesian_aeqd(
        lon, lat, radar.longitude["data"].item(), radar.latitude["data"].item()
    )


def plot_ppi_comparison(
    radar_orig,
    radar_filt,
    qtys,
    outdir=Path("."),
    ext="png",
    markers=None,
    gatefilter=None,
    max_dist=100,
    mask=None,
    mask_alpha=0.2,
):
    """Plot comparison of filtered and unfiltered PPIs."""
    cbar_ax_kws = {
        "width": "3%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.05, 0.0, 1, 1),
        "borderpad": 0,
    }

    ncols = 2
    n_qtys = len(qtys)
    nrows = n_qtys

    fig = plt.figure(figsize=(5 * ncols + 0.4, 5 * nrows), constrained_layout=True)
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)

    display_orig = pyart.graph.RadarDisplay(radar_orig)
    display_filt = pyart.graph.RadarDisplay(radar_filt)
    time = datetime.strptime(
        radar_orig.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
    )
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")

    if markers is not None:
        markers = [
            (*get_radar_coords(float(m[0]), float(m[1]), radar_orig), m[2], m[3])
            for m in markers
        ]

    for qi, qty in enumerate(qtys):
        axs = subfigs[qi, 0].subplots(
            nrows=1, ncols=ncols, squeeze=True, sharey=True, sharex=True
        )
        cax = inset_axes(axs[-1], bbox_transform=axs[-1].transAxes, **cbar_ax_kws)

        if "VRAD" in qty:
            plot_utils.QTY_RANGES[qty] = (
                -1 * np.ceil(radar_orig.get_nyquist_vel(0)),
                np.ceil(radar_orig.get_nyquist_vel(0)),
            )

        cmap, norm = plot_utils.get_colormap(qty)
        cbar_ticks = None

        if norm is None:
            # define the bins and normalize
            bounds = np.linspace(
                plot_utils.QTY_RANGES[qty][0], plot_utils.QTY_RANGES[qty][1], 40
            )
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(cmap, len(bounds))
        elif isinstance(norm, mpl.colors.BoundaryNorm):
            cbar_ticks = norm.boundaries

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(plot_utils.QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            ticks=cbar_ticks,
        )
        cbar.set_label(label=plot_utils.COLORBAR_TITLES[qty], weight="bold")

        for radar, display, ax in zip(
            [radar_orig, radar_filt], [display_orig, display_filt], axs.flat
        ):

            if qty == "HCLASS":
                plot_utils.set_HCLASS_cbar(cbar)

                # Set 0-values as nan to prevent them from being plotted with same color as 1
                radar.fields["radar_echo_classification"]["data"].set_fill_value(np.nan)

                radar.radar_fields["radar_echo_classification"][
                    "data"
                ] = np.ma.masked_values(
                    radar.radar_fields["radar_echo_classification"]["data"], 0
                )

            display.plot(
                plot_utils.PYART_FIELDS[qty],
                0,
                title="",
                ax=ax,
                axislabels_flag=False,
                colorbar_flag=False,
                cmap=cmap,
                norm=norm,
                zorder=10,
                gatefilter=gatefilter,
            )

            if mask is not None and "mask" in radar.fields.keys():
                display.plot(
                    mask,
                    0,
                    title="",
                    ax=ax,
                    axislabels_flag=False,
                    colorbar_flag=False,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    zorder=11,
                    alpha=mask_alpha,
                    raster=True,
                    # edgecolors="none",
                    gatefilter=gatefilter,
                )

            if "excluded" in radar.fields.keys():
                display.plot(
                    "excluded",
                    0,
                    title="",
                    ax=ax,
                    axislabels_flag=False,
                    colorbar_flag=False,
                    cmap="Pastel2_r",
                    vmin=1,
                    vmax=1,
                    zorder=11,
                    alpha=1.0,
                    raster=True,
                    mask_outside=True,
                    # edgecolors="none",
                    gatefilter=gatefilter,
                )

            if markers is not None:
                for marker in markers:
                    ax.plot(
                        marker[0] / 1000,
                        marker[1] / 1000,
                        marker=marker[3],
                        markerfacecolor="none",
                        markeredgecolor=marker[2],
                        zorder=20,
                    )

            for r in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
                display.plot_range_ring(r, ax=ax, lw=0.5, col="k")
            # display.plot_grid_lines(ax=ax, col="grey", ls=":")
            ax.set_title(plot_utils.TITLES[qty], y=-0.08)

        # x-axis
        for ax in axs.flat:
            ax.set_xlabel("Distance from radar (km)")
            ax.set_title(ax.get_title(), y=-0.22)
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlim([-max_dist, max_dist])
            ax.set_ylim([-max_dist, max_dist])
            ax.set_aspect(1)
            ax.grid(zorder=15, linestyle="-", linewidth=0.4)

        axs[0].set_title("Original data")
        axs[1].set_title("Filtered data")
        # y-axis
        # for ax in axs.flat:
        axs[0].set_ylabel("Distance from radar (km)")
        axs[0].yaxis.set_major_formatter(fmt)

    # # Remove empty axes
    # for ax in axes.flat[n_qtys:]:
    #     ax.remove()

    fig.set_constrained_layout_pads(wspace=0.005)
    fig.suptitle(
        f"{radar.metadata['instrument_name'].decode().capitalize()} "
        f"{time:%Y/%m/%d %H:%M} UTC {radar.fixed_angle['data'][0]:.1f}°",
        y=1.05,
    )

    if ext != "none":
        fname = outdir / (
            f"radar_vars_{radar.metadata['instrument_name'].decode().lower()[:3]}_"
            f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'][0]:.1f}.{ext}"
        )

        fig.savefig(
            fname,
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
        plt.close(fig)
        del fig
    else:
        return fig


def plot_ppi_fig(
    radar,
    qtys,
    ncols=2,
    max_dist=100,
    outdir=Path("."),
    ext="png",
    markers=None,
    title_prefix="",
    gatefilter=None,
    mask=None,
):
    """Plot a figure of 4 radar variables.

    Parameters
    ----------
    radar : pyart.core.Radar
        The radar object.
    qtys : list of str
        List of radar variables to plot.
    ncols : int, optional
        Number of columns, by default 2
    max_dist : int, optional
        Maximum distance to plot, by default 100
    outdir : pathlib.Path , optional
        Output path, by default Path(".")
    ext : str, optional
        Figure filename extension, by default "png"
    markers : list, optional
        List of marker tuples as (lon, lat, marker_color, marker_symbol), by default None
    title_prefix : str, optional
        Prefix to add to the figure title, by default ""
    gatefilter : pyart.filters.GateFilter, optional
        Gate filter object, by default None
    mask : str, optional
        Mask field name, by default None

    """
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    n_qtys = len(qtys)
    nrows = np.ceil(n_qtys / ncols).astype(int)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    display = pyart.graph.RadarDisplay(radar)
    time = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")

    if markers is not None:
        markers = [
            (*get_radar_coords(float(m[0]), float(m[1]), radar), m[2], m[3])
            for m in markers
        ]

    for ax, qty in zip(axes.flat, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        if "VRAD" in qty:
            plot_utils.QTY_RANGES[qty] = (
                -1 * np.ceil(radar.get_nyquist_vel(0)),
                np.ceil(radar.get_nyquist_vel(0)),
            )

        cmap, norm = plot_utils.get_colormap(qty)
        cbar_ticks = None

        if norm is None:
            # define the bins and normalize
            bounds = np.linspace(
                plot_utils.QTY_RANGES[qty][0], plot_utils.QTY_RANGES[qty][1], 40
            )
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(cmap, len(bounds))
        elif isinstance(norm, mpl.colors.BoundaryNorm):
            cbar_ticks = norm.boundaries

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(plot_utils.QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            ticks=cbar_ticks,
        )
        cbar.set_label(label=plot_utils.COLORBAR_TITLES[qty], weight="bold")

        if qty == "HCLASS":
            plot_utils.set_HCLASS_cbar(cbar)

            # Set 0-values as nan to prevent them from being plotted with same color as 1
            radar.fields["radar_echo_classification"]["data"].set_fill_value(np.nan)

            radar.fields["radar_echo_classification"]["data"] = np.ma.masked_values(
                radar.fields["radar_echo_classification"]["data"], 0
            )

        display.plot(
            plot_utils.PYART_FIELDS[qty],
            0,
            title="",
            ax=ax,
            axislabels_flag=False,
            colorbar_flag=False,
            cmap=cmap,
            norm=norm,
            zorder=10,
            gatefilter=gatefilter,
        )

        if mask is not None:
            display.plot(
                mask,
                0,
                title="",
                ax=ax,
                axislabels_flag=False,
                colorbar_flag=False,
                cmap="gray",
                vmin=0,
                vmax=1,
                zorder=11,
                alpha=0.2,
                raster=True,
                # edgecolors="none",
                gatefilter=gatefilter,
            )

        if markers is not None:
            for marker in markers:
                ax.plot(
                    marker[0] / 1000,
                    marker[1] / 1000,
                    marker=marker[3],
                    markerfacecolor="none",
                    markeredgecolor=marker[2],
                    zorder=20,
                )

        for r in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
            display.plot_range_ring(r, ax=ax, lw=0.5, col="k")
        # display.plot_grid_lines(ax=ax, col="grey", ls=":")
        ax.set_title(plot_utils.TITLES[qty], y=-0.08)

    # x-axis
    for ax in axes[1][:].flat:
        ax.set_xlabel("Distance from radar (km)")
        ax.set_title(ax.get_title(), y=-0.22)
        ax.xaxis.set_major_formatter(fmt)

    # y-axis
    for ax in axes.flat[::2]:
        ax.set_ylabel("Distance from radar (km)")
        ax.yaxis.set_major_formatter(fmt)

    for ax in axes.flat:
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_aspect(1)
        ax.grid(zorder=15, linestyle="-", linewidth=0.4)

    # Remove empty axes
    for ax in axes.flat[n_qtys:]:
        ax.remove()

    fig.suptitle(
        f"{title_prefix}{time:%Y/%m/%d %H:%M} UTC {radar.fixed_angle['data'][0]:.1f}°",
        y=1.02,
    )

    if ext != "none":
        fname = outdir / (
            f"radar_vars_{radar.metadata['instrument_name'].decode().lower()[:3]}_"
            f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'][0]:.1f}.{ext}"
        )

        fig.savefig(
            fname,
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
        plt.close(fig)
        del fig
    else:
        return fig, axes
