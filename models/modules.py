"""Descriptor network building blocks.

Only the modules required by :class:`~models.matcher.CrossViewMatcher` are
kept; attribute names (``resblocks`` / ``att_layer`` / ``final_block``)
match the released checkpoints verbatim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_layers.transformer import Transformer_self_att


def desc_l2norm(desc):
    """L2-normalise descriptors along the channel dimension (dim=1).

    Args:
        desc: [B, C, H, W] descriptor map.
    Returns:
        [B, C, H, W] with unit-norm descriptors per spatial position.
    """
    return F.normalize(desc, p=2, dim=1, eps=1e-10)


class BasicBlock(nn.Module):
    """Standard ResNet basic block: Conv3x3-BN-ReLU-Conv3x3-BN + shortcut.

    Args:
        in_planes: input channel count.
        planes: output channel count (``expansion`` is 1).
        stride: stride of the first convolution (1 = no downsampling).
        bn: use BatchNorm (False replaces it with Identity).
        padding_mode: convolution padding mode ('zeros').
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        # 1x1 projection when stride or channel count changes.
        self.shortcut = (
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                      stride=stride, bias=False)
            if stride != 1 or in_planes != self.expansion * planes
            else nn.Identity()
        )

    def forward(self, x, relu=True):
        """Apply the residual block.

        Args:
            x: [B, in_planes, H, W] input feature map.
            relu: apply a final ReLU. Set False on descriptor output layers so
                descriptors may take negative values.
        Returns:
            [B, planes, H/stride, W/stride] output feature map.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out) if relu else out


class DeepResBlockDesc(nn.Module):
    """Deep residual descriptor projector (DINO feature -> descriptor space).

    Pipeline: 3 residual blocks for channel compression (1024->512->256->128),
    a 3-layer self-attention transformer for spatial context, then a final
    residual block to ``last_dim`` channels with optional L2 normalisation.

    Args:
        bn: use BatchNorm in the residual blocks.
        last_dim: output descriptor channel count (usually 128).
        in_channels: input channel count (DINO feature dim, usually 1024).
        block_dims: intermediate channel sizes; the first three are used.
        add_posEnc: add sinusoidal position encoding before the attention
            layers (True for the perspective ground images, False for BEV
            satellite features).
        norm_desc: L2-normalise output descriptors so that dot products equal
            cosine similarities.
        padding_mode: convolution padding mode.
    """

    def __init__(self, bn, last_dim, in_channels, block_dims, add_posEnc,
                 norm_desc, padding_mode='zeros'):
        super().__init__()
        self.norm_desc = norm_desc

        self.resblocks = nn.Sequential(
            BasicBlock(in_channels, block_dims[0], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[0], block_dims[1], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[1], block_dims[2], bn=bn, padding_mode=padding_mode),
        )

        # d_model must match block_dims[2] (the post-compression channels).
        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

        # Linear output (relu=False in forward): descriptors need signed values.
        self.final_block = BasicBlock(block_dims[2], last_dim, bn=bn, padding_mode=padding_mode)

    def forward(self, x):
        """Project a dense feature map to (optionally normalised) descriptors.

        Args:
            x: [B, in_channels, H, W] input feature map.
        Returns:
            [B, last_dim, H, W] descriptor map (L2-normalised when
            ``norm_desc`` is True).
        """
        x = self.resblocks(x)
        x = self.att_layer(x)
        x = self.final_block(x, relu=False)
        return desc_l2norm(x) if self.norm_desc else x
