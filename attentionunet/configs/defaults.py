"""key default settings."""
import ml_collections
import torch
from paths import OUTPUT_PATHS, TRAINING_DATA_PATH

def get_default_configs():
    config = ml_collections.ConfigDict()

    # base directory for logs, checkpoints, and evaluation outputs
    config.base_output_dir = OUTPUT_PATHS

    # numeric precision used during training
    # supported values: "32", "16", "bf16"
    config.precision = "32"
    config.residuals = None

    #training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 16
    training.n_epochs = 100

    # optional buffer around prediction boundaries when computing loss
    # kept disabled as it did not improve results
    training.loss_buffer_width = None

    training.save_n_epochs = None

    #data
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 128

    # path to dataset containing coarse inputs and fine targets
    data.dataset_path = TRAINING_DATA_PATH

    # dataset splits for training and validation
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}

    # whether patches are sampled from variable spatial locations
    data.variable_location = False

    # optional fixed geographic region selection
    data.location_config = None

    # deprecated temporal conditioning option
    data.time_inputs = False

    # required only if inference uses a different lat/lon grid than training
    data.training_coords = None

    #model
    config.model = model = ml_collections.ConfigDict()

    # improved conditioning strategy (recommended default)
    model.cascade_conditioning = False

    # optional learnable location encodings (number of channels or None)
    model.location_parameters = None

    # deprecated side-channel conditioning
    model.side_conditioning = False

    model.ema_rate = None

    # attention implementation
    # "legacy": full global attention
    # "local": windowed attention
    # "rope": rotary positional encoding
    model.attention_type = "legacy"

    # base channel width controlling overall model capacity
    model.nf = 128

    # channel multipliers per UNet resolution level
    model.ch_mult = (1, 2, 2, 2)

    # number of residual blocks at each resolution
    model.num_res_blocks = 4

    # resolutions at which attention is applied
    model.attn_resolutions = (16,)

    model.resamp_with_conv = True
    model.conditional = True
    model.dropout = 0.1
    model.nonlinearity = "silu"

    # embedding used for noise-level (sigma) conditioning
    model.embedding_type = "fourier"

    model.diffusion = True

    #optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    # runtime device selection
    config.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
