import os
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

    if config["only_inside_mask"]:
        try:
            mask_field = radar.fields[config["mask_field_name"]]["data"]
            mask_field.set_fill_value(1)
            mask = mask & (~mask_field.filled().astype(bool))
        except KeyError:
            pass

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_gates(mask)


    try:
        previous_excluded = radar.fields["excluded"]["data"]
    except KeyError:
        previous_excluded = np.zeros_like(mask)

    radar.add_filter(
        gatefilter,
        replace_existing=True,
    )

    excluded = previous_excluded | gatefilter.gate_excluded
    radar.add_field_like("mask", "excluded", excluded, replace_existing=True)
    return radar, gatefilter


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
            radar, _ = filter(radar, filter_kwargs)
    return radar


def create_filters_from_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        filterconfig = config["filters"]
    filters = [(data_filter, filterconfig[k]) for k in filterconfig.keys()]
    return filters
