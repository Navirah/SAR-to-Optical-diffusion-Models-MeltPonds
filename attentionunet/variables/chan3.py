from diffusion_downscaling.data.scaling import Scalers

input_variables = [
'coarse_relative_humidity',
 'coarse_elevation',
 'coarse_surface_pressure']

output_variables = ['precipitation', 'coarse_relative_humidity',
 'coarse_elevation']

VARIABLE_SCALER_MAP = {
    "precipitation": Scalers.LOG_SCALER,
    "coarse_T200": Scalers.Z_SCALER,
    "coarse_T500": Scalers.Z_SCALER,
    "coarse_T700": Scalers.Z_SCALER,
    "coarse_T850": Scalers.Z_SCALER,
    "coarse_Q200": Scalers.Z_SCALER_NO_MEAN,
    "coarse_Q500": Scalers.Z_SCALER_NO_MEAN,
    "coarse_Q700": Scalers.Z_SCALER_NO_MEAN,
    "coarse_Q850": Scalers.Z_SCALER_NO_MEAN,
    "coarse_vorticity500": Scalers.Z_SCALER,
    "coarse_vorticity700": Scalers.Z_SCALER,
    "coarse_vorticity850": Scalers.Z_SCALER,
    "coarse_relative_humidity": Scalers.MIN_MAX_SCALER,
    "coarse_elevation": Scalers.Z_SCALER,
    "coarse_surface_pressure": Scalers.Z_SCALER,
}

VARIABLES = (input_variables, output_variables)
