import sys
from ..variables.all_var_T import VARIABLES, VARIABLE_SCALER_MAP
from defaults import get_default_configs
import torch

def get_config():
    config = get_default_configs()

    # ensure location_parameter_config exists for backward compatibility
    if not hasattr(config.model, "location_parameter_config"):
        config.model.location_parameter_config = getattr(
            config.model, "location_parameters", None
        )

    #training
    #config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.run_name = "multipredict/karras_p_diff_noscale"
    config.project_name = "diffusion_downscaling"
    config.model_type = "diffusion"
    config.diffusion_type = "karras"
    config.precision = "bf16"

    config.model.dropout = 0.05
    config.model.ema_rate = 0.999
    config.model.attn_resolutions = (4,)
    config.model.attention_resolutions = (4,)

    # where all experiment outputs will be written
    config.base_output_dir = "/path/to/outputs"

    training = config.training
    training.batch_size = 2
    training.n_epochs = 100
    training.loss_weights = [1.0]

    #data (TFRecord pipeline)
    data = config.data

    # variables must stay ordered — build_model relies on variables[1] for outputs
    # e.g. (["s1_vv","s1_vh"], ["s2_b2","s2_b3","s2_b4","s2_b8"])
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP

    # kept for compatibility, even though TFRecords handle resizing internally
    data.centered = True
    data.image_size = 256

    # enable TFRecord loading
    config.tfrecord = True

    # root directory containing train/ and val/ TFRecord folders
    data.tfrecord_root = "/path/to/tfrecords/256"

    # channel and size metadata expected by TFRecordDataset
    data.input_channels  = 2   # S1 VV,VH
    data.output_channels = 4   # S2 B2,B3,B4,B8
    data.tilesize        = 256
    data.img_height      = 256
    data.img_width       = 256

    # optional augmentations handled inside TFRecordDataset
    data.random_resize_factor = 1.0
    data.random_rotate = False

    # legacy NetCDF fields (ignored when tfrecord=True)
    data.dataset_path = "IGNORED_IN_TFRECORD_MODE.nc"
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices  = {"split": [0.9, 1]}

    #optim
    optim = config.optim
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 2e-4
    optim.grad_clip = 1.0
    optim.lr_schedule = {
        "cosine": {
            "t_initial": config.training.n_epochs,
            "lr_min": 1e-6
        }
    }

    return config
