"""Cross-view matching head.

Single :class:`CrossViewMatcher` serves all four datasets; state_dict keys
(``dustbin_score``, ``grd_projector.*``, ``sat_projector.*``) match the
released checkpoints verbatim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DeepResBlockDesc

# Channel schedule inside DeepResBlockDesc: 1024 -> 512 -> 256 -> 128 -> desc_dim.
_DEFAULT_BLOCK_DIMS = (512, 256, 128, 64)


class CrossViewMatcher(nn.Module):
    """Dense ground/satellite matcher with dustbin-augmented dual softmax.

    The matcher projects DINO feature maps of both views into a shared
    descriptor space, computes a cosine score matrix (scaled by
    ``temperature``), masks invalid ground pixels to -inf, and normalises
    with a dustbin-augmented dual softmax so only mutually-consistent pairs
    score highly.

    Args:
        device: torch device the module lives on.
        sat_bev_res: target spatial resolution of the satellite BEV feature
            map (features are bilinearly resized to ``sat_bev_res``^2).
        temperature: score scaling before the dual softmax; smaller is sharper.
        embed_dim: input feature channel count (DINO output, 1024).
        desc_dim: output descriptor channel count (128).

    Forward args:
        grd_feature: [B, embed_dim, H_grd, W_grd] ground DINO feature map.
        sat_feature: [B, embed_dim, H_sat, W_sat] satellite DINO feature map.
        mask: [B, N_grd] bool, True where the ground pixel is valid
            (depth-valid AND semantic mask); invalid columns are set to -inf
            before the dual softmax.

    Forward returns:
        matching_score: [B, N_sat, N_grd] dual-softmax matching probabilities
            in [0, 1] (input to pose solving / sampling).
        matching_score_original: [B, N_sat, N_grd] masked cosine scores
            (input to InfoNCE).
        sat_desc: [B, desc_dim, N_sat] satellite descriptors.
        grd_desc: [B, desc_dim, N_grd] ground descriptors.
    """

    def __init__(self, device, sat_bev_res, temperature=0.1, embed_dim=1024,
                 desc_dim=128):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.sat_bev_res = sat_bev_res

        # Learnable dustbin score absorbs probability mass of unmatched points.
        self.dustbin_score = nn.Parameter(torch.tensor(1.0))

        # Ground images are perspective -> positional encoding is required.
        self.grd_projector = DeepResBlockDesc(
            True,
            last_dim=desc_dim,
            in_channels=embed_dim,
            block_dims=_DEFAULT_BLOCK_DIMS,
            add_posEnc=True,
            norm_desc=True,
        )

        # Satellite BEV is already a metric grid -> no positional encoding.
        self.sat_projector = DeepResBlockDesc(
            True,
            last_dim=desc_dim,
            in_channels=embed_dim,
            block_dims=_DEFAULT_BLOCK_DIMS,
            add_posEnc=False,
            norm_desc=True,
        )

    def forward(self, grd_feature, sat_feature, mask):
        grd_desc = self.grd_projector(grd_feature).flatten(2)  # [B, desc_dim, N_grd]

        if (sat_feature.shape[2] != self.sat_bev_res
                or sat_feature.shape[3] != self.sat_bev_res):
            sat_feature = F.interpolate(
                sat_feature,
                (self.sat_bev_res, self.sat_bev_res),
                mode="bilinear",
                align_corners=False,
            )

        sat_desc = self.sat_projector(sat_feature).flatten(2)  # [B, desc_dim, N_sat]

        matching_score_original = torch.matmul(
            sat_desc.transpose(1, 2).contiguous(), grd_desc
        ) / self.temperature  # [B, N_sat, N_grd]

        # Mask invalid ground columns to -inf (probability 0 after softmax).
        matching_score_original = matching_score_original.masked_fill(
            ~mask.unsqueeze(1), float("-inf")
        )

        # Dustbin-augmented score matrix: [B, N_sat+1, N_grd+1].
        batch_size, num_sat, num_grd = matching_score_original.shape
        sat_dustbin = self.dustbin_score.expand(batch_size, num_sat, 1)
        grd_dustbin = self.dustbin_score.expand(batch_size, 1, num_grd)
        dustbin_corner = self.dustbin_score.expand(batch_size, 1, 1)
        couplings = torch.cat(
            [
                torch.cat([matching_score_original, sat_dustbin], dim=-1),
                torch.cat([grd_dustbin, dustbin_corner], dim=-1),
            ],
            dim=1,
        )

        # Dual softmax enforces mutual consistency between the two views.
        couplings = F.softmax(couplings, dim=1) * F.softmax(couplings, dim=2)

        matching_score = couplings[:, :-1, :-1]  # strip dustbin row/column
        return matching_score, matching_score_original, sat_desc, grd_desc
