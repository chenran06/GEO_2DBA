"""Training losses: InfoNCE matching, BEV pose-consistency and the
GT-pose cross_unnorm objective (monitoring only)."""
from losses.infonce import compute_infonce_loss_match_all_with_scale_select_negatives
from losses.bev_cross import loss_bev_cross
from losses.cross_gt import cross_unnorm_loss

__all__ = [
    "compute_infonce_loss_match_all_with_scale_select_negatives",
    "loss_bev_cross",
    "cross_unnorm_loss",
]
