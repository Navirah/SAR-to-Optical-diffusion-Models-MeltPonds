"""A custom python wrapper of CDO using subprocess to execute cdo on the command line.

A simple python wrapper for running the most import cdo processing steps,
in particular remapcon and mergetime. 

We ran into issues memory limits, so added a batched functionality
that runs multiple cdo commands in parallel before merging
everything together.
"""

import subprocess
from tqdm import tqdm
from pathlib import Path
import pickle

import xarray as xr
import numpy as np

from typing import NamedTuple
import joblib


def create_parallel_executor(num_workers):
    return lambda input_list, function: joblib.Parallel(n_jobs=num_workers)(
        joblib.delayed(function)(input) for input in input_list
    )


class LatLon(NamedTuple):
    latlon: tuple[float, float]
    latlon_sizes: tuple[int, int]
    latlon_incs: tuple[float, float]


class GridCoordinateManager:
    """Utility for computing latlon data required for remapcon.

    Generates the lat and longitude starting points, increments, 
    and grid sizes using precomputed lat lon grid_data.
    """

    def __init__(self, grid_data_path):
        with open(grid_data_path, "rb") as f:
            self.lat_lon_grid = pickle.load(f)

    def get_nearest_gridpoints(self, lat, lon, clip_lower: bool):
        recentered_lon = self.lat_lon_grid[1] - 180
        lat_diff = self.lat_lon_grid[0] - lat
        lon_diff = recentered_lon - lon
        lat_idx = np.where(lat_diff > 0)[0][0]
        lon_idx = np.where(lon_diff > 0)[0][0]
        if clip_lower:
            lat_idx = lat_idx - 1
            lon_idx = lon_idx - 1
        lonlat = self.lat_lon_grid[0][lat_idx], recentered_lon[lon_idx]
        ids = (lat_idx, lon_idx)
        return lonlat, ids

    def compute_latlon_data(self, grid_coords, grid_overshoot=True):
        starting_points = self.get_nearest_gridpoints(
            grid_coords[2], grid_coords[0], clip_lower=grid_overshoot
        )
        ending_points = self.get_nearest_gridpoints(
            grid_coords[3], grid_coords[1], clip_lower=not grid_overshoot
        )

        grid_sizes = (
            ending_points[1][0] - starting_points[1][0] + 1,
            ending_points[1][1] - starting_points[1][1] + 1,
        )
        incs = (
            abs(self.lat_lon_grid[0][0] - self.lat_lon_grid[0][1]),
            abs(self.lat_lon_grid[1][0] - self.lat_lon_grid[1][1]),
        )
        latlon_data = LatLon(starting_points[0], grid_sizes, incs)
        return latlon_data


class CDOWrapper:

    variable_renaming_map = {
        "era5_temperature": ("t", "T"),
        "era5_specific": ("q", "Q"),
        "era5_vorticity": ("__xarray_dataarray_variable__", "vorticity"),
    }

    def __init__(self, num_workers=None, batch_size=None):
        self.num_workers = num_workers
        self.batch_size = batch_size

    def execute(self, command):
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        return result

    def select_regions_and_merge_batched(
        self, netcdf_files, coordinates, out_file: Path
    ):
        """Selects the desired region in batches and then merges all data across time.
        """
        if self.batch_size is None:
            self.batch_size = len(netcdf_files)
        iterator = range(0, len(netcdf_files), self.batch_size)
        if self.num_workers is None:
            batched_outputs = []
            for i in tqdm(iterator, desc="Selecting and merging batches"):
                batch_output_file = self._apply_select_merge_to_batch(
                    netcdf_files, coordinates, out_file, self.batch_size, i
                )

                batched_outputs.append(batch_output_file)
        else:
            executor = create_parallel_executor(self.num_workers)
            batched_outputs = executor(
                iterator,
                lambda i: self._apply_select_merge_to_batch(
                    netcdf_files, coordinates, out_file, self.batch_size, i
                ),
            )

        merge_command = f"cdo -O -mergetime {' '.join(batched_outputs)} {out_file}"
        res = self.execute(merge_command)
        if res.returncode != 0:
            return res

        for file in batched_outputs:
            Path(file).unlink()
        return res

    def _apply_select_merge_to_batch(
        self, netcdf_files, coordinates, out_file, batch_size, i
    ):
        batch_output_file = f'{out_file.with_suffix("")}_{i}.nc'
        try:
            rename_tuple = self._get_renaming_tuple(out_file)
        except KeyError:
            rename_tuple = None
        batch = netcdf_files[i : i + batch_size]
        files_string = " ".join([str(file) for file in batch])
        self.select_regions_and_merge(
            files_string, coordinates, batch_output_file, rename_tuple
        )
        return batch_output_file

    def _get_renaming_tuple(self, out_file_name):
        split_str = out_file_name.with_suffix("").name.split("_")
        variable_name = "_".join(split_str[:2])
        xr_var_names = self.variable_renaming_map[variable_name]
        try:
            elevation = int(split_str[-1])
        except ValueError:
            elevation = None

        new_var_name = (
            xr_var_names[1] if elevation is None else f"{xr_var_names[1]}{elevation}"
        )
        rename_tuple = (xr_var_names[0], new_var_name)
        return rename_tuple

    def select_regions_and_merge(
        self, in_file_pattern, coordinates, out_file, rename_tuple
    ):

        temp_file_coarsened = f"temp_{Path(out_file).name}"
        command = ""
        coords_string = ",".join([str(coord) for coord in coordinates])
        if rename_tuple is None:
            temp_file_coarsened = out_file

        command += f"cdo -O -mergetime -apply,-sellonlatbox,{coords_string} [ {in_file_pattern} ] {temp_file_coarsened}\n"
        if rename_tuple is not None:
            command += f"cdo -chname,{rename_tuple[0]},{rename_tuple[1]} {temp_file_coarsened} {out_file}\n"

        command += f"cdo sinfo {out_file}\n"
        res = self.execute(command)
        if rename_tuple is not None:
            Path(temp_file_coarsened).unlink()

        return res

    def resample_and_combine(
        self,
        in_file,
        out_file,
        latlon_data: LatLon,
        reselected_coords,
        temp_file="temp.nc",
    ):

        coords_string = ",".join([str(coord) for coord in reselected_coords])
        conservative_remapping_config = self.create_remapcon_config_file(*latlon_data)
        parameter_table_path = self.create_renamed_variables_table(in_file)
        temp_file_coarsened = "temp_renamed.nc"
        temp_file_reselected = "temp_renamed_2.nc"

        command = ""
        # command += f'ncatted -O -a units,lat,c,c,"degrees north" -a units,lon,c,c,"degrees east" {in_file}' # need these attrs for remapcon
        command += (
            f"cdo -O -sellonlatbox,{coords_string} {in_file} {temp_file_reselected}\n"
        )
        command += (
            f"cdo -O remapcon,{conservative_remapping_config} {in_file} {temp_file}\n"
        )
        command += f"cdo -O setpartabn,{parameter_table_path},convert {temp_file} {temp_file_coarsened}\n"
        command += f"cdo -O -merge [  {temp_file_reselected} {temp_file_coarsened} ] {out_file}\n"
        command += f"cdo sinfo {out_file}\n"

        res = self.execute(command)

        for file in [temp_file, temp_file_coarsened, temp_file_reselected]:
            Path(file).unlink()

        return res

    def get_names(self, in_file):
        names = list(xr.open_dataset(in_file).data_vars.keys())
        return names

    def create_renamed_variables_table(
        self, in_file, filepath="temp_variable_table.txt"
    ):

        variable_names = self.get_names(in_file)
        renamed_variables = [f"coarse_{var}" for var in variable_names]

        with open(filepath, "w") as f:
            for old, new in zip(variable_names, renamed_variables):
                f.write("&parameter\n")
                f.write(f"name\t={old}\n")
                f.write(f"out_name\t={new}\n")
        return filepath

    def create_remapcon_config_file(
        self, start_coords, sizes, incs, filepath="temp_remapcon_config.txt"
    ):

        file_string = f"""\
gridtype  = lonlat
gridsize  = {sizes[0]*sizes[1]}
xsize     = {sizes[1]}
ysize     = {sizes[0]}
xname     = lon
xlongname = "longitude"
xunits    = "degrees_east"
yname     = lat
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = {start_coords[1]}
xinc      = {incs[1]}
yfirst    = {start_coords[0]}
yinc      = {incs[0]}
        """
        # CESM xinc = 1.25, y_inc = 0.942408376963351
        with open(filepath, "w") as f:
            f.write(file_string)
        return filepath

    def merge(self, in_files, out_file):
        command = f"cdo -O merge  {' '.join(in_files)} {out_file}"
        return self.execute(command)

    def merge_with_wildcard(self, in_files, out_file, variable_wildcard):

        coarse_vars = sum(
            [
                self._select_variables_with_wildcard(file, variable_wildcard)
                for file in in_files
            ],
            [],
        )

        command = f"cdo -O -merge -apply,-selvar,{','.join(coarse_vars)} [ {' '.join(in_files)} ] {out_file}"
        return self.execute(command)

    def _select_variables_with_wildcard(self, in_file, variable_wildcard):
        variables = self.get_names(in_file)
        return [var for var in variables if variable_wildcard in var]
