"""Data loading ML utils to seamlessly load xrarray data.

PyTorch provides highly-optimised data loading functionality, 
which we can make use of by creating custom PyTorch Datasets.

These require a __getitem__ method, which should load in a single
input / output pair of data from the provided xarrays. PyTorch then
takes care of all of the CPU multiprocessing for loading in the data
efficiently, and batching to pass inputs into the model in parallel.

Some of this code (XRDataset) was adapted from https://github.com/henryaddison/mlde. 

This file defines a range of Dataset custom classes for loading in data
in various ways, as well as some utilities for building these datasets
and PyTorch dataloaders.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pandas as pd
import sys
sys.path.append("/home/nkamal/projects/colocation/diffusion_model_alex/src/diffusion_downscaling/data")
from scaling import LatLonScaler
from itertools import product

from .constants import LOCATIONS_MAP


def remove_leap_days(dates_list):

    dates_pd = pd.to_datetime(dates_list)
    mask = (dates_pd.month == 2) & (dates_pd.day == 29)

    # Apply mask to filter out leap years
    dates_pd_masked = dates_pd[~mask]

    # Convert back to numpy datetime64
    dates_np_masked = np.array(dates_pd_masked, dtype="datetime64[D]")
    return dates_np_masked


class XRDataset(Dataset):
    def __init__(self, ds, variables, target_variables, time_range):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range

    @classmethod
    def variables_to_tensor(cls, ds, variables):
        """
        Simple utility function to convert xarray dataset to tensor.
        :param ds: xarray dataset
        :param variables: list of variables to select from xarray dataset
        :return: torch.tensor
        """
        if len(variables) > 0:
            return torch.tensor(
                # stack features before lat-lon (HW)
                np.stack([ds[var].values for var in variables], axis=-3)
            ).float()
        return torch.tensor(np.array([]))

    @classmethod
    def time_to_tensor(cls, ds, shape, time_range):
        """
        Currently unused.
        """
        climate_time = np.array(ds["time"] - time_range[0]) / np.array(
            [time_range[1] - time_range[0]], dtype=np.dtype("timedelta64[ns]")
        )
        season_time = ds["time.dayofyear"].values / 360

        return (
            torch.stack(
                [
                    torch.tensor(climate_time).broadcast_to(
                        (climate_time.shape[0], *shape[-2:])
                    ),
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                ],
                dim=-3,
            )
            .squeeze()
            .float()
        )

    def __len__(self):
        return len(self.ds.time)

    def sel(self, idx):
        return self.ds.isel(time=idx)


class UpsampleXRDataset(XRDataset):
    """
    Upsample predictors to match lat-lon resolution of target variables.
    We modify the variables_to_tensor method to interpolate the selected variables to the target resolution.
    Uses nearest neighbor interpolation; bicubic interpolation didn't seem to improve performace.

    """

    def select_ds(self, ds, variables):
        return ds[variables]

    def variables_to_tensor_upsample(self, ds, variables):
        # select variables
        selected_ds = self.select_ds(ds, variables)
        selected_ds = selected_ds.interp(
            lat_2=ds.lat.values, lon_2=ds.lon.values, method="nearest"
        )

        return self.variables_to_tensor(selected_ds, variables)

    def __getitem__(self, idx):
        """
        Standard getitem method for pytorch dataset.
        We select the dataset at the given index, convert the variables to tensor,
        and return the condition (predictor) and target (predictand) variables.
        """
        subds = self.sel(idx)

        cond = self.variables_to_tensor_upsample(subds, self.variables)
        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)

        time = subds["time"].values.reshape(-1)

        return cond, x, [], time

def find_nearest(array, value):
    # find index of nearest value in array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class VariableLocationXRDataset(UpsampleXRDataset):
    """
    When training over large domains, we randomly select subpatches of the domain
    for training. This class selects random lat-lon coordinate slices and returns
    them as the condition for the model.
    """

    def __init__(self, ds, variables, target_variables, time_range, image_size=128, training_coords = None):
        super().__init__(ds, variables, target_variables, time_range)

        self.image_size = image_size
        self.latlonscaler = LatLonScaler(ds, image_size)

        if training_coords is None:
            self.train_lats, self.train_lons = self.ds.lat.values, self.ds.lon.values
        else:
            self.train_lats, self.train_lons = training_coords

    def _select_coordinates(self):
        """
        Simple approach to selecting random lat/lon slices over the domain.
        latlonscaler uses the image_size to determine the possible lat/lon slices.
        """
        lat_idx = np.random.randint(0, len(self.latlonscaler.possible_lats))
        lon_idx = np.random.randint(0, len(self.latlonscaler.possible_lons))
        lats = self.train_lats[slice(lat_idx, lat_idx + self.image_size)]
        lons = self.train_lons[slice(lon_idx, lon_idx + self.image_size)]

        return lats, lons

    def get_coord_anchor_and_lengths(self, subds):
        """
        In-case the model uses location-specific conditional information,
        we compute the top-left lat-lon index and the length of the lat-lon slices.
        We refer to this is the coord anchor and lengths.

        These must be computed with respect to the training lat lon grids. 
        """
        lats, lons = subds.lat.values, subds.lon.values
        lat_idx = find_nearest(self.train_lats, lats[0])
        lon_idx = find_nearest(self.train_lons, lons[0])
        return np.array([[lat_idx, lon_idx], [len(lats), len(lons)]])

    def __getitem__(self, idx):
        subds = self.sel(idx)

        coords = self._select_coordinates()

        subds = subds.sel(lat=coords[0], lon=coords[1])
        processed_coords = self.get_coord_anchor_and_lengths(subds)
        cond = self.variables_to_tensor_upsample(subds, self.variables)

        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)

        time = subds["time"].values.reshape(-1)

        return (cond, processed_coords), x, coords, time


class FixedLocationXRDataset(VariableLocationXRDataset):
    """
    VariableLocationXRDataset but with fixed latitude and longitude coordinates.

    This inherits all methods from VariableLocationXRDataset except for overriding
    the _select_coordinates method, which just returns a fixed lat lon tuple.

    :param ds: scaled xarray to load predictors and predictands
    :param variables: list, predictor variable names
    :param target_variables: list, output variable names
    :param fixed_latlon: tuple, fixed latitude and longitude coordinates to select from ds.
    """

    def __init__(self, ds, variables, target_variables, time_range, fixed_latlon, training_coords=None):
        super().__init__(ds, variables, target_variables, time_range, training_coords=training_coords)

        self.fixed_latlon = fixed_latlon

    def _select_coordinates(self):
        """Simple override to return a fixed lat-lon slice pair."""
        return self.fixed_latlon



from torch.utils.data import default_collate


def custom_collate(batch):

    return (
        *default_collate([(e[0], e[1]) for e in batch]),
        np.concatenate([e[2] for e in batch]),
        np.concatenate([e[3] for e in batch]),
    )


def select_custom_coordinates(
    xr_data, variable_location_config, loss_buffer_width=None
):
    try:
        if (
            isinstance(variable_location_config, str)
            and variable_location_config in LOCATIONS_MAP.keys()
        ):
            lats, lons = LOCATIONS_MAP[variable_location_config]
        else:
            lats, lons = variable_location_config
    except:
        print('Invalid location config')
        print('Continuing over whole domain')
        return xr_data.lat.values, xr_data.lon.values
    start_lat_idx = find_nearest(xr_data.lat.values, lats[0])
    start_lon_idx = find_nearest(xr_data.lon.values, lons[0])
    end_lat_idx = find_nearest(xr_data.lat.values, lats[1]) + 1
    end_lon_idx = find_nearest(xr_data.lon.values, lons[1]) + 1
    if loss_buffer_width is not None:
        start_lat_idx -= loss_buffer_width
        start_lon_idx -= loss_buffer_width
        end_lat_idx += loss_buffer_width
        end_lon_idx += loss_buffer_width
    return (
        xr_data.lat.values[start_lat_idx:end_lat_idx],
        xr_data.lon.values[start_lon_idx:end_lon_idx],
    )


def build_dataloader(
    xr_data,
    variables,
    target_variables,
    batch_size,
    shuffle,
    include_time_inputs,
    variable_location,
    variable_location_config,
    image_size,
    buffer_width,
    training_coords,
    num_workers,
):

    time_range = None
    if variable_location:
        if variable_location_config is None:
            xr_dataset = VariableLocationXRDataset(
                xr_data, variables, target_variables, time_range, image_size, training_coords
            )
        else:
            fixed_latlon = select_custom_coordinates(
                xr_data, variable_location_config, buffer_width
            )
            xr_dataset = FixedLocationXRDataset(
                xr_data, variables, target_variables, time_range, fixed_latlon, training_coords
            )
    else:
        xr_dataset = UpsampleXRDataset(xr_data, variables, target_variables, time_range)

    data_loader = DataLoader(
        xr_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader


def get_dataloader(
    active_dataset_name,
    variables,
    data_transform,
    variable_location,
    location_config,
    image_size,
    buffer_width,
    training_coords,
    batch_size,
    split,
    include_time_inputs=False,
    evaluation=False,
    shuffle=True,
    num_workers=1,
):
    """Create data loaders for given split.

    Args:
      active_dataset_name: Name of dataset from which to load data splits
      variables: 2-tuple of string lists ([input_variable_names], [output_variable_names])
      data_transform: Callable to transforms xarray dataset
      batch_size: Size of batch to use for DataLoaders
      split: Split of the active dataset to load
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      data_loader
    """

    xr_data = prepare_and_scale_data(active_dataset_name, split, location_config, data_transform)

    predictor_variables, target_variables = variables

    data_loader = build_dataloader(
        xr_data,
        predictor_variables,
        target_variables,
        batch_size,
        shuffle,
        include_time_inputs,
        variable_location,
        location_config,
        image_size,
        buffer_width,
        training_coords,
        num_workers,
    )

    return data_loader


def prepare_and_scale_data(active_dataset_name, split, variable_location_config, data_transform = None):

    xr_data = xr.open_dataset(active_dataset_name)
    coords = select_custom_coordinates(xr_data, variable_location_config)
    xr_data = xr_data.sel(lat=coords[0], lon=coords[1])
    try:
        xr_data = xr_data.assign_coords(time=xr_data.indexes["time"].to_datetimeindex())
    except:
        pass

    if isinstance(split, tuple):
        split, custom_date_range = split
        # should probably only do this for cesm eval
        custom_date_range = remove_leap_days(custom_date_range)
        indices = slice(*(split * len(xr_data.time)).astype(int))
        xr_data = xr_data.isel(time=indices).drop_duplicates("time")
        xr_data = xr_data.sel(time=custom_date_range)
    else:
        indices = slice(*(split * len(xr_data.time)).astype(int))
        xr_data = xr_data.isel(time=indices)

    if data_transform is not None:
        xr_data = data_transform(xr_data)
    return xr_data
