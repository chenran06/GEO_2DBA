"""Self-attention transformer over dense feature maps.

Keeps the exact module structure of the released checkpoints
(``att_layer.layers.{0,1,2}`` plus the (non-persistent) positional buffer).
"""
import copy
import math

import torch
import torch.nn as nn
from einops.einops import rearrange

from .transformer_utils import EncoderLayer


class PositionEncodingSine(nn.Module):
    """2D sinusoidal position encoding generalised to rectangular images."""

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            d_model: feature dimension (channels).
            max_shape: maximum feature-map size the buffer covers.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2))
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        # Non-persistent: recomputed at construction, absent from state_dict.
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """Add positional encoding. Args/returns: [N, C, H, W]."""
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class Transformer_self_att(nn.Module):
    """Stack of self-attention encoder layers over a dense feature map.

    Args:
        d_model: feature dimension (must equal the input channel count).
        num_layers: number of stacked :class:`EncoderLayer` blocks.
        add_posEnc: add :class:`PositionEncodingSine` before attention.
    """

    def __init__(self, d_model, num_layers, add_posEnc=False):
        super().__init__()

        self.d_model = d_model
        layer_names = ['self'] * num_layers
        attention = 'linear'
        self.nheads = 8
        self.layer_names = layer_names
        encoder_layer = EncoderLayer(d_model, self.nheads, attention)
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()
        self.add_posEnc = add_posEnc
        self.posEnc = PositionEncodingSine(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats):
        """Run self-attention over the feature map.

        Args:
            feats: [N, d_model, H, W] dense feature map.
        Returns:
            [N, d_model, H, W] context-enhanced feature map.
        """
        assert self.d_model == feats.size(1), \
            "The feature size and transformer must be equal"

        b, c, h, w = feats.size()

        if self.add_posEnc:
            feats = self.posEnc(feats)

        feats = rearrange(feats, 'n c h w -> n (h w) c')

        for layer, name in zip(self.layers, self.layer_names):
            feats = layer(feats, feats)

        feats = feats.transpose(2, 1).reshape((b, c, h, w))
        return feats
