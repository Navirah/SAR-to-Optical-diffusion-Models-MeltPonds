COLORADO_RIVER_BASIN_COORDS_WIDE = [-116, -100, 30, 46]
COLORADO_MSWEP_COORDS = [-114.3, -101.5, 31.3, 44.1]


# Some latitude / longitude preset ranges to run evaluation over prespecified
# regions. These are used in the sampling script configs to specify the regions to
# extract predictors and predictands from.
LOCATIONS_MAP = {
    "colorado": ([44.05, 31.349998], [-114.24999, -101.549995]),
}
# can also use your own coord arrays if desired
TRAINING_COORDS_LOOKUP = {}