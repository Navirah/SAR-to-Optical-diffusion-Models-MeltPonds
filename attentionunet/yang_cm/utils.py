from .unet import UNetModel
import torch.nn as nn

nonlinearity_opts = {'silu': nn.SiLU, 'lrelu': nn.LeakyReLU}


def create_model(config):

    nf = config.model.nf
    ch_mult = config.model.ch_mult
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    embedding_type = config.model.embedding_type.lower()
    resamp_with_conv = config.model.resamp_with_conv
    image_size = config.data.image_size
    if getattr(config, "tfrecord", False):
        cond_var_channels = int(config.data.input_channels)    # e.g., 2 for S1 VV,VH
        output_channels   = int(config.data.output_channels)   # e.g., 4 for S2 B2,B3,B4,B8
    else:
        cond_var_channels, output_channels = list(map(len, config.data.variables))
    diffusion = config.model.diffusion
    conditional = config.model.conditional
    side_conditioning = config.model.side_conditioning
    cascade_conditioning = config.model.cascade_conditioning
    location_parameters = config.model.location_parameter_config
    attention_type = config.model.attention_type
    nonlinearity = config.model.nonlinearity


    model = UNetModel(
        image_size,
        cond_var_channels,
        nf,
        output_channels,
        num_res_blocks,
        attn_resolutions,
        cascade_conditioning=cascade_conditioning,
        side_conditioning=side_conditioning,
        embedding_type=embedding_type,
        location_parameters=location_parameters,
        attention_type=attention_type,
        dropout=dropout,
        channel_mult=ch_mult,
        conv_resample=resamp_with_conv,
        diffusion=diffusion,
        conditional=conditional,
        nonlinearity=nonlinearity_opts[nonlinearity]
    )

    return model
