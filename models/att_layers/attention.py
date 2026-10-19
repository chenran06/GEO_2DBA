"""Attention primitives shared by the descriptor transformer blocks.

`Attention` supports three backends selected by the ``attention`` argument:
``'linear'`` (ELU feature-map linear attention, the default used by
``Transformer_self_att``), ``'full'`` (scaled dot-product) and ``'flash'``.
"""
import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F


def elu_feature_map(x):
    """ELU+1 feature map for linear attention (always positive, invertible)."""
    return torch.nn.functional.elu(x) + 1


class Attention(Module):
    """Multi-head attention dispatcher.

    Args:
        eps: numerical stabiliser for the linear-attention denominator.
        use_dropout: apply dropout on the attention weights.
        attention: one of ``'linear'``, ``'full'``, ``'flash'``.
        attention_dropout: dropout probability on attention weights.
    """

    def __init__(self, eps=1e-6, use_dropout=False, attention='', attention_dropout=0.1):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)
        self.attention = attention

    def forward_full(self, queries, keys, values):
        """Multi-head scaled dot-product attention.

        Args:
            queries: [N, L, H, D]
            keys:    [N, S, H, D]
            values:  [N, S, H, D]
        Returns:
            queried_values: [N, L, H, D]
        """
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        softmax_temp = 1.0 / queries.size(3) ** 0.5
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)
        return queried_values.contiguous()

    def forward_linear(self, queries, keys, values):
        """Linear attention from "Transformers are RNNs" (Katharopoulos et al.).

        Args and returns are shaped as in :meth:`forward_full`.
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        return queried_values.contiguous()

    def forward_flash(self, q, k, v):
        """Half-precision flash attention (PyTorch SDPA)."""
        args = [x.half().contiguous() for x in [q, k, v]]
        v = F.scaled_dot_product_attention(*args).to(q.dtype)
        return v.contiguous()

    def forward(self, queries, keys, values):
        """Dispatch to the configured attention backend. Returns [N, L, H, D]."""
        if self.attention == 'linear':
            x = self.forward_linear(queries, keys, values)
        elif self.attention == 'flash':
            x = self.forward_flash(queries, keys, values)
        else:
            x = self.forward_full(queries, keys, values)
        return x
