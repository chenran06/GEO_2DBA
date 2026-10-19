"""Transformer encoder layer used by the descriptor self-attention blocks.

The layer is a pre-merge residual block: multi-head (linear) self/cross
attention followed by a feed-forward network over the concatenation of the
input and the attention message.
"""
import torch
import torch.nn as nn

from .attention import Attention


class EncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and an FFN.

    Args:
        d_model: feature dimension of the input tokens (default 128).
        nhead: number of attention heads.
        attention: attention backend for :class:`Attention`
            (``'linear'`` or ``'full'``).
    """

    def __init__(self, d_model, nhead, attention='linear'):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(attention=attention)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        """Attend ``x`` to ``source`` and return the residual-updated tokens.

        Args:
            x:      [N, L, C] query tokens.
            source: [N, S, C] key/value tokens.
        Returns:
            [N, L, C] updated tokens (``x + message``).
        """
        bs = x.size(0)
        query, key, value = x, source, source

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)

        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
