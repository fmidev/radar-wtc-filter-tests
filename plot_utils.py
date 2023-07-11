"""Utility functions."""
import os
import re
import struct
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyart
from cmcrameri import cm as cm_crameri
from matplotlib import cm, colors
from pyart.io.sigmet import SigmetFile

alias_names = {
    "VRAD": [
        "velocity",
        "radial_wind_speed",
        "corrected_velocity",
        "VRADDH",
    ],
    "DBZH": [
        "reflectivity",
        "DBZHC",
    ],
    "wind_shear": [
        "radial_shear",
        "azimuthal_shear",
    ],
    "total_shear": ["total_shear_thr", "tdwr_gfda_shear"],
    "ZDR": ["corrected_differential_reflectivity", "ZDRC"],
}


PYART_FIELDS = {
    "DBZH": "reflectivity",
    "DBZHC": "corrected_reflectivity",
    "HCLASS": "radar_echo_classification",
    "KDP": "specific_differential_phase",
    "PHIDP": "differential_phase",
    "RHOHV": "cross_correlation_ratio",
    "SQI": "normalized_coherent_power",
    "TH": "total_power",
    "VRAD": "velocity",
    "VRADDH": "corrected_velocity",
    "WRAD": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "ZDRC": "corrected_differential_reflectivity",
    "SNR": "signal_to_noise_ratio",
    "LOG": "log_signal_to_noise_ratio",
    "PMI": "polarimetric_meteo_index",
    "CSP": "clutter_power_ratio",
}

PYART_FIELDS_ODIM = {
    "DBZH": "reflectivity_horizontal",
    "DBZHC": "corrected_reflectivity",
    "HCLASS": "radar_echo_classification",
    "KDP": "specific_differential_phase",
    "PHIDP": "differential_phase",
    "RHOHV": "cross_correlation_ratio",
    "SQI": "normalized_coherent_power",
    "TH": "total_power_horizontal",
    "VRAD": "velocity",
    "VRADH": "velocity_horizontal",
    "WRAD": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "SNR": "signal_to_noise_ratio",
    "LOG": "log_signal_to_noise_ratio",
    "PMI": "polarimetric_meteo_index",
    "CSP": "clutter_power_ratio",
}


QTY_FORMATS = {
    # "reflectivity": "{x:.0f}",
    "DBZH": "{x:.0f}",
    "VRAD": "{x:.0f}",
    "VRADH": "{x:.0f}",
    # "radial_wind_speed": "{x:.0f}",
    "SNR": "{x:.0f}",
    "ZDR": "{x:.1f}",
    "RHOHV": "{x:.2f}",
    "KDP": "{x:.1f}",
    "HCLASS": "{x:1.0f}",
    "PHIDP": "{x:.2f}",
    "SQI": "{x:.2f}",
    "TH": "{x:.0f}",
    "WRAD": "{x:.1f}",
    "LOG": "{x:.0f}",
    "cnr": "{x:.0f}",
    "wind_shear": "{x:.0f}",
    # "radial_shear": "{x:.0f}",
    # "azimuthal_shear": "{x:.0f}",
    "total_shear": "{x:.0f}",
    # "total_shear_thr": "{x:.0f}",
    "tdwr_gfda": "{x:.0f}",
    "PMI": "{x:.2f}",
    "CSP": "{x:.0f}",
    "LOG": "{x:.0f}",
}


QTY_RANGES = {
    "DBZH": (-15.0, 60.0),
    "HCLASS": (1.0, 6.0),
    "KDP": (-4.0, 8.0),
    "PHIDP": (0, 360.0),
    "RHOHV": (0.8, 1.0),
    "SQI": (0.0, 1.0),
    "TH": (-15.0, 60.0),
    "VRAD": (-10.0, 10.0),
    "VRADH": (-10.0, 10.0),
    # "radial_wind_speed": (-30.0, 30.0),
    "WRAD": (0.0, 5.0),
    "ZDR": (-4, 5.0),
    "SNR": (-30.0, 50.0),
    "LOG": (0.0, 50.0),
    "cnr": (-30, 0),
    "wind_shear": (-20, 20),
    # "azimuthal_shear": (-20, 20),
    "total_shear": (0, 15),
    # "total_shear_thr": (0, 20),
    "tdwr_gfda": (-1, 1),
    "PMI": (0, 1),
    "CSP": (-20, 50),
    "LOG": (-20, 50),
}

COLORBAR_TITLES = {
    # "reflectivity": "Equivalent reflectivity factor [dBZ]",
    "DBZH": "Equivalent reflectivity factor [dBZ]",
    "HCLASS": "HydroClass",
    "KDP": "Specific differential phase [°/km]",
    "PHIDP": "Differential phase [°]",
    "RHOHV": "Copolar correlation coefficient",
    "SQI": "Normalized coherent power",
    "TH": "Total reflectivity factor [dBZ]",
    "VRAD": "Doppler velocity [m s$^{-1}$]",
    # "velocity": "Doppler velocity [m s$^{-1}$]",
    "VRADH": "Doppler velocity [m s$^{-1}$]",
    "WRAD": "Doppler spectrum width [m s$^{-1}$",
    "ZDR": "Differential reflectivity [dB]",
    "SNR": "Signal-to-noise ratio [dB]",
    "LOG": "LOG signal-to-noise ratio [dB]",
    # "radial_wind_speed": "Doppler velocity [m s$^{-1}$]",
    "cnr": "Carrier-to-noise ratio [dB]",
    "wind_shear": "Wind shear [m s$^{-1}$ km$^{-1}$]",
    # "azimuthal_shear": "Azimuthal wind shear [m s$^{-1}$ km$^{-1}$]",
    "total_shear": "Total wind shear [m s$^{-1}$ km$^{-1}$]",
    # "total_shear_thr": "",
    "tdwr_gfda": "TWDR gust front segments",
    "PMI": "Polarimetric meteo index",
    "CSP": "Clutter power ratio of dBT to -dBZ [dB]",
    "LOG": "Log receiver signal-to-noise ratio [dB]",
}

TITLES = {
    "DBZH": r"$Z_{e}$",
    "DBZHC": r"Corrected $Z_{e}$",
    "VRAD": r"$v$",
    "RHOHV": r"$\rho{HV}$",
    "ZDR": r"$Z_{DR}$",
    "PMI": r"PMI",
    "WRAD": r"$\sigma_v$",
    "PHIDP": r"$\Phi_{DP}$",
    "SNR": r"SNR",
    "HCLASS": "HydroClass",
    "KDP": r"$K_{DP}$",
    "SQI": "SQI",
    "TH": r"$Z_{t}$",
    "LOG": "LOG",
    "cnr": "CNR",
    "wind_shear": "Wind shear",
    "total_shear": "Wind shear",
    "tdwr_gfda": "Wind shear",
    "PMI": "PMI",
    "CSP": "CSP",
    "LOG": "LOG",
}

for name, alias_list in alias_names.items():
    for a in alias_list:
        if a not in COLORBAR_TITLES.keys():
            COLORBAR_TITLES[a] = COLORBAR_TITLES[name]
        if a not in QTY_RANGES.keys():
            QTY_RANGES[a] = QTY_RANGES[name]
        if a not in QTY_FORMATS.keys():
            QTY_FORMATS[a] = QTY_FORMATS[name]
        if a not in TITLES.keys():
            TITLES[a] = TITLES[name]


def get_colormap(quantity):
    if quantity == "HCLASS":
        cmap = colors.ListedColormap(
            ["peru", "dodgerblue", "blue", "cyan", "yellow", "red"]
        )
        norm = colors.BoundaryNorm(np.arange(0.5, 7.5), cmap.N)
    elif "VRAD" in quantity or quantity in alias_names["VRAD"]:
        # cmap = "pyart_BuDRd18"
        # cmap = cm_crameri.roma
        cmap = "cmc.roma_r"
        norm = None
    elif quantity in ["DBZH", *alias_names["DBZH"]]:
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 5)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
    elif quantity == "TH":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 5)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
    elif quantity in ["SNR", "LOG", "CSP"]:
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 5)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
    elif quantity == "KDP":
        cmap = "pyart_Theodore16"
        norm = None
    elif quantity == "PHIDP":
        cmap = "pyart_Wild25"
        norm = None
    elif quantity == "RHOHV":
        bounds = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1.05]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_RefDiff", len(bounds))
    elif "WRAD" in quantity:
        cmap = "pyart_NWS_SPW"
        norm = None
    elif quantity == "ZDR":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.1, 0.5)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_LangRainbow12", len(bounds))
    elif quantity == "cnr":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.1, 1.0)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("viridis", len(bounds))
    elif quantity == "PMI":
        bounds = np.arange(
            QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.01, 0.05
        )
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("cmc.hawaii", len(bounds))
    elif quantity == "SQI":
        bounds = np.arange(
            QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.01, 0.05
        )
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("cmc.hawaii", len(bounds))
    elif quantity in ["wind_shear", *alias_names["wind_shear"]]:
        cmap = "cmc.roma_r"
        norm = None
    elif quantity in ["total_shear", *alias_names["total_shear"]]:
        cmap = "cmc.hawaii_r"
        norm = None
    elif "total_shear_thr" in quantity:
        cmap = "tab20"
        norm = None
    elif "tdwr_gfda" == quantity:
        cmap = "tab20"
        norm = None
    else:
        cmap = cm.viridis
        norm = None

    return cmap, norm


def set_HCLASS_cbar(cbar):
    labels = get_HCLASS_labels()
    values = np.arange(1, 7)
    cbar.set_ticks(values, labels=labels)
    cbar.set_ticklabels(labels)
    plt.setp(
        cbar.ax.get_yticklabels(),
        rotation="vertical",
        va="center",
    )


def get_HCLASS_labels():
    return [
        "1 Other",
        "2 Rain",
        "3 WSnow",
        "4 Ice",
        "5 Graup",
        "6 Hail",
    ]


def get_sigmet_file_list_by_task(
    path, file_regex="WRS([0-9]{12}).RAW([A-Z0-9]{4})", task_name=None
):
    """Generate a list of files by task.

    Parameters
    ----------
    path : str
        Path to data location
    file_regex : str
        Regex for matching filenames.
    task_name : str
        If given, only return files matching this task.

    Returns
    -------
    out : dict
        Dictionary of lists of filenames in `path`, with key giving the task name.

    """
    # Browse files
    prog = re.compile(file_regex.encode().decode("unicode_escape"))
    fullpath = os.path.abspath(path)

    data = []

    for (root, dirs, files) in os.walk(fullpath):
        addpath = root.replace(fullpath, "")

        for f in files:
            result = prog.match(f)

            if result is None:
                continue

            try:
                # Get task name from headers
                sf = pyart.io.sigmet.SigmetFile(os.path.join(root, f))
                task = (
                    sf.product_hdr["product_configuration"]["task_name"]
                    .decode()
                    .strip()
                )
                sf.close()
                if task_name is not None and task != task_name:
                    continue
                data.append([os.path.join(addpath, f), task])
            except (ValueError, OSError):
                continue
            except struct.error:
                print(f"Failed to read {f} with pyart!")
                continue

    df = pd.DataFrame(data, columns=["filename", "task_name"])
    out = {}
    for task, df_task in df.groupby("task_name"):
        out[task] = df_task["filename"].to_list()

    return out


def parse_time_from_filename(fn, pattern, timepattern, group_idx=0):
    """Parse time from filename."""
    po = re.compile(pattern)
    match = po.match(fn)
    if match is not None:
        return datetime.strptime(match.groups()[group_idx], timepattern)
    else:
        return None


def add_estimated_SNR(fn, radar):
    sigmet_file = SigmetFile(fn)

    # Get reflectivity calibration value
    NEZ = (
        sigmet_file.ingest_header["task_configuration"]["task_calib_info"][
            "reflectivity_calibration"
        ]
        / 16
    )
    # Ranges in the same shape as data
    R = np.tile(radar.range["data"], (radar.nrays, 1)) * 1e-3
    # Estimated SNR
    SNR = radar.fields["reflectivity"]["data"] - NEZ - 20 * np.log10(R)

    field = {
        "units": "dB",
        "standard_name": "signal_to_noise_ratio",
        "long_name": "Signal to noise ratio",
        "coordinates": "elevation azimuth range",
        "data": SNR,
    }

    # Add a field
    radar.add_field("signal_to_noise_ratio", field)

    sigmet_file.close()
