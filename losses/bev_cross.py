"""BEV-space pose-consistency loss between the GT pose and the solved pose.

A reference BEV grid ``X0`` is transformed by the GT pose (C_gt, yaw_gt) and
by the solver output (C_opt, psi_opt, with ``psi = -yaw``); the loss is the
mean pointwise Euclidean distance between the two transformed point sets:

    R(yaw) = [[cos yaw, -sin yaw], [sin yaw, cos yaw]]   (GT)
    R(-psi) = [[cos psi, sin psi], [-sin psi, cos psi]]  (prediction)
    X_gt   = R(yaw_gt)  @ X0 + C_gt
    X_pred = R(-psi_opt) @ X0 + C_opt
    loss_b = mean_n || X_gt[b, n] - X_pred[b, n] ||

The cross_unnorm objective is 180-degree periodic in psi, so the solved
(psi, C) pair carries a 180-degree ambiguity; this loss intentionally keeps
a single rotation candidate (no 180-degree folding), matching the behaviour
of the original training script.
"""
import torch


def _rot_from_yaw(yaw):
    """[B] yaw angles -> [B, 2, 2] rotation matrices R(yaw)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1),
    ], dim=-2)


def _rot_from_psi(psi):
    """[B] psi angles -> [B, 2, 2] rotation matrices R(-psi)."""
    c, s = torch.cos(psi), torch.sin(psi)
    return torch.stack([
        torch.stack([c, s], dim=-1),
        torch.stack([-s, c], dim=-1),
    ], dim=-2)


def loss_bev_cross(X0, C_opt, psi_opt, C_gt, yaw_gt):
    """Mean pointwise distance between GT- and prediction-transformed grids.

    Args:
        X0:      [N, 2] or [B, N, 2] reference BEV points (metric).
        C_opt:   [B, 2] or [B, 1, 2] solved camera position (meters).
        psi_opt: [B] solved yaw in the solver convention (radians).
        C_gt:    [B, 2] or [B, 1, 2] GT camera position (meters).
        yaw_gt:  [B] GT yaw (radians).

    Returns:
        loss: [B] per-sample mean pointwise distance in meters.
    """
    if X0.dim() == 3:
        X0 = X0.reshape(-1, 2)
    if C_opt.dim() == 3:
        C_opt = C_opt.squeeze(1)
    if C_gt.dim() == 3:
        C_gt = C_gt.squeeze(1)

    R_gt = _rot_from_yaw(yaw_gt)
    R_pr = _rot_from_psi(psi_opt)

    X0b = X0.unsqueeze(0)
    X_gt = X0b @ R_gt.transpose(1, 2) + C_gt.unsqueeze(1)
    X_pr = X0b @ R_pr.transpose(1, 2) + C_opt.unsqueeze(1)

    return (X_gt - X_pr).norm(dim=-1).mean(dim=-1)
