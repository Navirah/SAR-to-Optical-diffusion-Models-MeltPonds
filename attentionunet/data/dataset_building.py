"""Range of utilities for loading and processing geospatial data in various formats.

This utilises our python cdo wrapper to apply the various processing steps
(resampling and merging) required to build our datasets. We implement 
this across different DataTypes, one for each of MSWEP, MSWX and ERA5.

DatasetBuilder then iterates over all the desired variables across
each of the datatypes to build the final netcdf file.
"""

from pathlib import Path
from .processing import CDOWrapper, LatLon

from abc import ABC, abstractmethod
from copy import deepcopy

from typing import NamedTuple, List


class GeographicData(NamedTuple):
    latlon_grid_data: LatLon
    coarse_coord_range: List
    sampled_coord_range: List


class DataType(ABC):
    """
    Abstract data type for merging and resampling all the required data types.
    """

    datatype_name = None

    def __init__(
        self,
        datetime_range: tuple,
        geographic_data: GeographicData,
        cdo_wrapper: CDOWrapper,
    ):
        first_year = datetime_range[0].year
        last_year = datetime_range[1].year
        self.years_in_datetime_range = list(map(str, range(first_year, last_year + 1)))

        self.geographic_data = geographic_data
        self.cdo = cdo_wrapper

    def create_coarsened_files(self, config, output_directory):
        """Key function for resampling and merging a given data type.

        Utilises our python cdo wrapper to perform batched region selection
        and merging, followed by conservative remapping and further merging.
        Each step is done in batches whose size is specified in the cdo wrapper.

        We expect child classes to fill in _get_config_iterator and
        _build_filename_list to tell us the variable names and paths
        under which to find the data.

        :param config: dict-like, contains information about where to find data files
        :param output_directory: str, base output directory for all processed files

        :return output_files: list, list of strings of output files to keep track of batches.
        """
        latlon, coarse_coords, sampled_coords = self.geographic_data

        config_iterator = self._get_config_iterator(config)
        output_files = []
        for variable, path in config_iterator:
            print(f"Beginning work on: {self.datatype_name} {variable}")
            output_file = Path(output_directory) / f"{self.datatype_name}_{variable}.nc"
            coarse_output_file = (
                Path(output_directory) / f"{self.datatype_name}_{variable}_coarse.nc"
            )
            filenames = self._build_filename_list(variable, path)

            res = self.cdo.select_regions_and_merge_batched(
                filenames, coarse_coords, output_file
            )
            if res.returncode != 0:
                print(res.returncode)
                print(res.stderr)

            res = self.cdo.resample_and_combine(
                output_file, coarse_output_file, latlon, sampled_coords
            )
            output_files.append((output_file, coarse_output_file))

        return output_files

    @abstractmethod
    def _get_config_iterator(self, config):
        pass

    @abstractmethod
    def _build_filename_list(self, variable, path):
        """Generate a list of paths for all the data for a given variable.

        Each dataset is stored slightly differently, so we need to adapt
        to the particular file structure used for MSWEP, MSWX, and ERA5. 

        :param variable: str, variable name when needed (i.e. not included in path)
        :param path: str, base path of the dataset to trawl for data.

        :return filelist: list, list of Path objects containing all the data
        """
        pass


class MSWEP_DataType(DataType):

    datatype_name = "mswep"

    def _get_config_iterator(self, config):
        return config.items()

    def _build_filename_list(self, variable, path):
        mswep_files = sum(
            [
                list(Path(path).rglob(f"*{year}*.nc"))
                for year in self.years_in_datetime_range
            ],
            [],
        )
        return mswep_files


class MSWX_DataType(DataType):

    datatype_name = "mswx"

    def _get_config_iterator(self, config):
        return config.items()

    def _build_filename_list(self, variable, path):
        mswx_files = sum(
            [
                list(Path(path).rglob(f"*{year}*.nc"))
                for year in self.years_in_datetime_range
            ],
            [],
        )
        return mswx_files


class ERA5_DataType(DataType):

    datatype_name = "era5"

    def _get_config_iterator(self, config):
        era_5_path, variable_dict = config

        return {key: era_5_path for key in variable_dict.keys()}.items()

    def _build_filename_list(self, variable, path):
        mswx_files = sum(
            [
                list(Path(path).rglob(f"{year}/*{variable}*.nc"))
                for year in self.years_in_datetime_range
            ],
            [],
        )
        return mswx_files


from typing import List, Tuple, Type


class DatasetBuilder:
    """Highest-level class for processing all data types.

    Given the date range, geographic data, and the cdo wrapper,
    this class can then take the data configuration to build datasets in stages.
    """

    def __init__(
        self,
        datetime_range: tuple,
        geographic_data: GeographicData,
        cdo_wrapper: CDOWrapper,
    ):
        self.datetime_range = datetime_range
        self.geographic_data = geographic_data
        self.cdo = cdo_wrapper

    def build_coarsened_files(
        self, data_type_config_map: List[Tuple[Type[DataType], tuple]], output_directory
    ):
        """Process all requested data variables according to their datatypes.

        We take a list of data configurations, each specifying the data type and
        the associated path / variable parameters, and create resampled files for 
        all of them.

        :param data_type_config_map: list, data types with their associated variable
            configurations
        :param output_directory: str, base output directory to save all processed files

        :return output_files: list, all final processed output file names. 
        """

        output_directory.mkdir(parents=True, exist_ok=True)
        output_files = []
        for data_type_class, config in data_type_config_map:
            data_type = data_type_class(
                self.datetime_range, self.geographic_data, self.cdo
            )
            output_files += data_type.create_coarsened_files(config, output_directory)

        return output_files

    def cleanup_non_coarse_files(self, output_directory):
        all_files = list(output_directory.glob('*.nc'))
        filtered = [file for file in all_files if 'coarse' not in str(file)]
        for file in filtered:
            file.unlink()