""" Scaling utilities for diffusion predictors and predictands.

A range of scaling utilities; these generally scale the data between -1, 1. 
This range is kept as convention from EDM (Karras et al.) since normal diffusion
requires its outputs to be scaled a certain way. We also scale inputs within the same
range to allow training to proceed effectively. 
"""

from enum import Enum
import numpy as np
import json
import pickle


class Scalers(str, Enum):
    """Most should be self-explanatory.

    Z-scaling (aka standard scaling, normal scaling).
    Min-max scaler scales min/max to -1/1
    Threshold scaler excludes certain values below threshold from std calculation.
    Log-scaling scales between 0 and 1.
    """

    Z_SCALER = 'z_scaler'
    Z_SCALER_NO_MEAN = 'z_scaler_no_mean'
    MIN_MAX_SCALER = 'min_max'
    Z_SCALER_THRESHOLD = 'z_scaler_threshold'
    NO_SCALER = 'no_scaler'
    LOG_SCALER = 'log_scaler'
    SQRT_SCALER = 'sqrt_scaler'


class No_Scaler:
    def fit(self, ds):
        pass

    def transform(self, data):
        return data.copy()

    def inverse_transform(self, data):
        return data.copy()


class Z_Scaler:

    def fit(self, ds):
        self.means = ds.mean().values
        self.stds = ds.std().values * 5

    def transform(self, data):
        data = data.copy()
        scaled_data = (data - self.means) / self.stds

        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        unscaled_data = (data * self.stds) + self.means
        return unscaled_data


class Z_Scaler_No_Mean:

    def fit(self, ds):
        self.stds = ds.std().values * 5

    def transform(self, data):
        data = data.copy()
        scaled_data = data / self.stds
        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        unscaled_data = (data) * self.stds
        return unscaled_data


class Z_Scaler_Threshold(Z_Scaler):

    def __init__(self, threshold=-np.inf, custom_scale=5):
        self.threshold = threshold
        self.custom_scale = custom_scale

    def fit(self, ds):
        filtered_ds = ds.where(ds > self.threshold, drop=True)
        self.means = filtered_ds.mean().values
        self.stds = filtered_ds.std().values * self.custom_scale

    # 0.5 to move things between 0-1, which works better
    # for the Karras diffusion model setup (sigma_data etc.)
    def transform(self, data):
        data = data.copy()
        # scaled_data = ((data - self.means) / self.stds) + 0.5
        scaled_data = (data - self.means) / self.stds
        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        # unscaled_data = ((data - 0.5) * self.stds) + self.means
        unscaled_data = ((data) * self.stds) + self.means
        return unscaled_data


class Min_Max_Scaler:

    def fit(self, ds):
        # fully reduce to a single mean and std
        self.mins = ds.min().values
        self.maxs = ds.max().values

    def transform(self, data):
        data = data.copy()
        scaled_data = (((data - self.mins) / (self.maxs - self.mins)) - 1) * 2
        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        unscaled_data = (data / 2 + 1) * (self.maxs - self.mins) + self.mins
        return unscaled_data


class Log_Scaler:
    def __init__(self, normalise_scale=2):
        self.scale = normalise_scale

    def fit(self, ds):
        pass

    def transform(self, data):
        data = data.copy()
        scaled_data = np.log10(1 + data) / self.scale
        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        unscaled_data = 10 ** (self.scale * data) - 1
        return unscaled_data


class Sqrt_Scaler:
    def __init__(self, normalise_scale=15):
        self.scale = normalise_scale

    def fit(self, ds):
        pass

    def transform(self, data):
        data = data.copy()
        scaled_data = np.sqrt(data) / self.scale
        return scaled_data

    def inverse_transform(self, data):
        data = data.copy()
        unscaled_data = (self.scale * data) ** 2
        return unscaled_data


class DataScaler:
    """Generic xarray datascaler, with a fit call and transformation functions.
    """

    enum_to_scaler = {
        Scalers.Z_SCALER: Z_Scaler,
        Scalers.Z_SCALER_NO_MEAN: Z_Scaler_No_Mean,
        Scalers.MIN_MAX_SCALER: Min_Max_Scaler,
        Scalers.Z_SCALER_THRESHOLD: Z_Scaler_Threshold,
        Scalers.NO_SCALER: No_Scaler,
        Scalers.LOG_SCALER: Log_Scaler,
        Scalers.SQRT_SCALER: Sqrt_Scaler,
    }

    def __init__(self, variable_scaler_map: dict):

        self.variable_scaler_map = {
            var: self.enum_to_scaler[scaler_enum]()
            for var, scaler_enum in variable_scaler_map.items()
        }

    def fit(self, ds):
        for var in self.variable_scaler_map.keys():
            self.variable_scaler_map[var].fit(ds[var])

    def transform(self, ds, exclude_vars=[]):
        ds = ds.copy()
        for var in self.variable_scaler_map.keys():
            if var not in exclude_vars:
                ds[var] = self.variable_scaler_map[var].transform(ds[var])
        return ds

    def inverse_transform(self, ds):
        ds = ds.copy()
        for var in self.variable_scaler_map.keys():
            ds[var] = self.variable_scaler_map[var].inverse_transform(ds[var])
        return ds

    def set_transform_exclusion(self, exclusion_vars):
        from functools import partial
        self.transform = partial(self.transform, exclude_vars=exclusion_vars)

    def to_dict(self):
        scaler_to_enum = {v.__name__:k for k,v in self.enum_to_scaler.items()}
        return {var : (json.dumps(scaler_to_enum[scaler.__class__.__name__]),scaler.__dict__) for var, scaler in self.variable_scaler_map.items()}
    
    def save_scaler_parameters(self, output_path):
        parameter_dict = self.to_dict()
        with open(output_path, 'wb') as fout:
            pickle.dump(parameter_dict, fout)

    def load_scaler_parameters(self, parameter_path):
        print(f'Loading scaler parameters from {parameter_path}')
        with open(parameter_path, 'rb') as fin:
            parameter_dict = pickle.load(fin)
        self.load_dict(parameter_dict)

    def load_dict(self, scaler_dict):
        self.variable_scaler_map = {}
        for var, (scaler_type, attrs) in scaler_dict.items():
            scaler = self.enum_to_scaler[json.loads(scaler_type)]()
            for key, value in attrs.items():
                setattr(scaler, key, attrs[key])
            self.variable_scaler_map[var] = scaler
class LatLonScaler:

    def __init__(self, ds, image_size):
        half_size = image_size // 2
        self.half_size = half_size
        self.possible_lats = ds.lat.values[half_size:-half_size]
        self.possible_lons = ds.lon.values[half_size:-half_size]

    def transform(self, coords):
        lat_norm = (coords[0] - self.possible_lats[0]) / (
            self.possible_lats[-1] - self.possible_lats[0]
        )
        lon_norm = (coords[1] - self.possible_lons[0]) / (
            self.possible_lons[-1] - self.possible_lons[0]
        )
        return lat_norm, lon_norm

    def inverse_transform(self, coords):
        lat_norm = (
            coords[0] * (self.possible_lats[-1] - self.possible_lats[0])
            + self.possible_lats[0]
        )
        lon_norm = (
            coords[1] * (self.possible_lons[-1] - self.possible_lons[0])
            + self.possible_lons[0]
        )
        return lat_norm, lon_norm
