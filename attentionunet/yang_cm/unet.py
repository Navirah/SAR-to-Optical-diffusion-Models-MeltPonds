from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class SplitConvDownsample(nn.Module):
    """
    A downsampling layer combined with convolution.

    :param channels: channels in the inputs and outputs.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)

        self.conv = conv_nd(
            dims, self.channels, self.out_channels, 3, stride=1, padding=1
        )

        self.downscale = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        downsample = self.downscale(x)
        return downsample, self.conv(downsample)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        num_embeddings=1,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        act=nn.SiLU,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.num_embeddings = num_embeddings
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            act(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embedders = nn.ModuleList([])
        for _ in range(num_embeddings):
            emb_layers = nn.Sequential(
                act(),
                linear(
                    emb_channels,
                    (
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                ),
            )
            self.embedders.append(emb_layers)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            act(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.num_embeddings > 0:
            emb_out = th.stack(
                [
                    emb_layers(emb[:, i]).type(h.dtype)
                    for i, emb_layers in enumerate(self.embedders)
                ],
                dim=0,
            )  # .to(h.device)
            emb_out = th.sum(emb_out, dim=0)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="legacy",
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
        image_size=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention_type = attention_type
        if attention_type == "flash":
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, encoder_out=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(
                self.attention, (qkv,), (), self.use_attention_checkpoint
            )
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        from flash_attn import flash_attn_qkvpacked_func

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = partial(
            flash_attn_qkvpacked_func, dropout_p=attention_dropout
        )
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv = self.inner_attn(
            qkv,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange

        self.rearrange = rearrange

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        : N is batchsize, T is number of tokens, H heads and C channels.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # qkv = qkv.half()

        qkv = self.rearrange(  # s is now number of tokens
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads
        )
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs * self.n_heads, ch, length)
        k = k.reshape(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == 2 * ch * self.n_heads
            ek, ev = encoder_kv.chunk(2, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.W = nn.Parameter(th.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        #x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        W = self.W.to(device=x.device, dtype=x.dtype)
        x_proj = x[:, None] * W[None, :] * 2 * np.pi
        return th.cat([th.sin(x_proj), th.cos(x_proj)], dim=-1)


class LocationFourierProjection(GaussianFourierProjection):

    def forward(self, x):
        lat, lon = x[:, 0], x[:, 1]
        lat_proj = lat[:, None] * self.W[None, : self.embedding_size // 2] * 2 * np.pi
        lon_proj = lon[:, None] * self.W[None, self.embedding_size // 2 :] * 2 * np.pi
        return th.cat(
            [
                th.sin(lat_proj),
                th.cos(lat_proj),
                th.sin(lon_proj),
                th.cos(lon_proj),
            ],
            dim=-1,
        )


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=th.float32, device="cpu"
):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            return th.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (th.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(
                3 * variance
            )
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "normal")



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        embedding_type="fourier",
        cascade_conditioning=False,
        side_conditioning=False,
        location_parameters=None,
        attention_type="legacy",
        nonlinearity=nn.SiLU,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        diffusion=True,
        conditional=True
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.cascade_conditioning = cascade_conditioning
        self.side_conditioning = side_conditioning
        self.conditional = conditional
        num_embeddings = 0
        if conditional:
            num_embeddings +=1
        if self.side_conditioning:
            num_embeddings +=1
        if diffusion:
            self.in_channels += out_channels
        self.location_parameters = location_parameters
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.embedding_type = embedding_type
        time_embed_dim = model_channels * 4

        if conditional:
            if embedding_type == "fourier":
                self.timestep_embedding = GaussianFourierProjection(
                    embedding_size=model_channels, scale=1
                )
                temb_input_dim = 2 * model_channels
            elif embedding_type == "positional":
                self.timestep_embedding = partial(timestep_embedding, dim=model_channels)
                temb_input_dim = model_channels
            elif embedding_type == "identity":
                self.timestep_embedding = nn.Identity()
                temb_input_dim = model_channels
            else:
                raise ValueError(f"embedding type {embedding_type} unknown.")

            self.time_embed = nn.Sequential(
                linear(temb_input_dim, time_embed_dim),
                nonlinearity(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.side_conditioning:
            self.location_embedding = LocationFourierProjection(
                embedding_size=model_channels, scale=1
            )
            temb_input_dim = 2 * model_channels
            self.loc_embed = nn.Sequential(
                linear(temb_input_dim, time_embed_dim),
                nonlinearity(),
                linear(time_embed_dim, time_embed_dim),
            )

        conditional_channels = self.in_channels - (
            self.out_channels if diffusion else 0
        )
        if cascade_conditioning:
            self.cascade_downscalers = nn.ModuleList(
                [
                    SplitConvDownsample(
                        conditional_channels, dims=dims, out_channels=32
                    )
                    for i, _ in enumerate(channel_mult)
                ]
            )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        train_size = self.image_size
        for level, mult in enumerate(channel_mult):
            for res_idx in range(num_res_blocks):
                if self.cascade_conditioning and res_idx == 0 and level > 0:
                    ch = ch + 32
                else:
                    ch = ch
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        num_embeddings=num_embeddings,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        act=nonlinearity
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            attention_type=attention_type,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            image_size=train_size,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            num_embeddings=num_embeddings,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            act=nonlinearity
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                train_size = train_size // 2
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                num_embeddings=num_embeddings,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                act=nonlinearity
            ),
            AttentionBlock(
                ch,
                attention_type=attention_type,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                image_size=train_size,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                num_embeddings=num_embeddings,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                act=nonlinearity
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        num_embeddings=num_embeddings,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        act=nonlinearity
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            attention_type=attention_type,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            image_size=train_size,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            num_embeddings=num_embeddings,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            act=nonlinearity
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                    train_size = train_size * 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nonlinearity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    # @th.compile()
    def forward(self, x, cond=None, timesteps=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs (the current sample/latent).
        :param cond: either a Tensor of conditioning channels, or (cond_tensor, coords).
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # ---- 1) Robustly unpack cond ----
        cond_tensor, coords = None, None
        if self.side_conditioning:
            # expected: cond is (cond_tensor, coords)
            cond_tensor, coords = cond
        elif isinstance(cond, (list, tuple)) and len(cond) == 2 and th.is_tensor(cond[0]):
            # sometimes it's passed as a tuple even if side_conditioning is False
            cond_tensor, coords = cond
        else:
            # plain tensor or None
            cond_tensor = cond if th.is_tensor(cond) else None

        # ---- 2) Concatenate image-like conditioner with x (if present) ----
        if cond_tensor is not None and x is not None:
            x = th.cat([x, cond_tensor], dim=1)
        elif x is None and cond_tensor is not None:
            x = cond_tensor  # fall back to conditioner alone if x is None

        # Keep a working variable named 'cond' for cascade downscalers later
        cond = cond_tensor

        # ---- 3) Early channel sanity check vs model expectation ----
        if x is None:
            raise ValueError("UNet forward received x=None and cond=None; nothing to process.")
        if x.dim() < 4:
            raise ValueError(f"UNet expects NCHW 4D input; got shape {tuple(x.shape)}.")
        in_ch_expected = self.in_channels
        in_ch_actual = x.shape[1]
        if in_ch_actual != in_ch_expected:
            raise RuntimeError(
                f"[UNet] Channel mismatch before first conv: got {in_ch_actual} channels, "
                f"but model was built for {in_ch_expected}. "
                f"Likely cause: missing/extra concatenation of the conditional tensor.\n"
                f" - tip: if diffusion=True, UNet in_channels = cond_channels + out_channels."
            )

        # ---- 4) Build timestep (and optional location) embeddings ----
        hs = []
        emb = None
        if self.conditional:
            # ensure timesteps track x's device/dtype (training-safe)
            t_device, t_dtype = x.device, x.dtype
            if not th.is_tensor(timesteps):
                timesteps = th.as_tensor(timesteps, device=t_device, dtype=t_dtype)
            else:
                timesteps = timesteps.to(device=t_device, dtype=t_dtype)

            if self.embedding_type == "fourier":
                timesteps = th.log(timesteps)

            # raw timestep embedding
            emb_in = self.timestep_embedding(timesteps)

            # inference-only: hard-match the time-MLP weights' device/dtype to avoid addmm mismatches
            if not self.training:
                tew = next(self.time_embed.parameters())
                emb_in = emb_in.to(device=tew.device, dtype=tew.dtype, non_blocking=True)

            emb = self.time_embed(emb_in)

            if self.side_conditioning:
                # align coords to time-MLP only in inference (training flow unchanged)
                if not self.training and coords is not None:
                    coords = coords.to(device=tew.device, dtype=tew.dtype, non_blocking=True)
                loc_emb = self.loc_embed(self.location_embedding(coords))
                emb = th.stack([emb, loc_emb], dim=1)
            else:
                emb = emb.unsqueeze(dim=1)

        # ---- 5) UNet body ----
        h = x.type(x.dtype)
        downscaling_idx = 0
        for idx, module in enumerate(self.input_blocks):
            if self.cascade_conditioning and (idx - 1) % 5 == 0 and idx > 2:
                # cascade uses the image-like conditioner
                cond, conv_cond = self.cascade_downscalers[downscaling_idx](cond)
                downscaling_idx += 1
                h = th.cat([h, conv_cond], dim=1)
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)
