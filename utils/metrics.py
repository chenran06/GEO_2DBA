"""Orientation-error metrics shared by training and evaluation.

Conventions: ``psi`` is the solver yaw in radians with ``psi = -yaw`` where
``yaw`` is the dataset GT heading in radians. Errors are reported in degrees
with circular wrapping and an optional 180-degree-ambiguity fold.
"""
import numpy as np


def circ_diff_deg(a_deg, b_deg):
    """Elementwise circular difference of two angle arrays, in [0, 180].

    Args:
        a_deg, b_deg: arrays of angles in degrees.
    Returns:
        np.ndarray of min(|a-b| mod 360, 360 - |a-b| mod 360).
    """
    d = np.abs(a_deg - b_deg) % 360.0
    return np.minimum(d, 360.0 - d)


def diff_yaw(psi_pred, psi_gt, fold_180=True):
    """Per-sample circular yaw error in degrees.

    Args:
        psi_pred: [B] predicted yaw (radians, solver convention).
        psi_gt:   [B] ground-truth yaw (radians, ``psi = -yaw`` convention).
        fold_180: fold errors > 90 degrees to their 180-degree complement
            (use True when comparing against 180-degree-ambiguous solvers).
    Returns:
        np.ndarray [B] of per-sample errors in degrees.
    """
    psi_pred_deg = np.rad2deg(psi_pred.detach().cpu().numpy())
    psi_gt_deg = np.rad2deg(psi_gt.detach().cpu().numpy())

    err = circ_diff_deg(psi_pred_deg, psi_gt_deg)
    if fold_180:
        err = np.minimum(err, circ_diff_deg(psi_pred_deg, psi_gt_deg + 180.0))
    return err


def compute_yaw_errors(rotation_gt, rotation_pred):
    """Per-sample yaw error from SO(2) rotation matrices (Procrustes branch).

    Extracts angles via ``arctan2(R[1,0], R[0,0])`` and compares against GT
    with circular wrapping (matching the legacy evaluation behaviour).

    Args:
        rotation_gt:   [B, 2, 2] ground-truth rotation matrices.
        rotation_pred: [B, 2, 2] predicted rotation matrices.
    Returns:
        list[float] of per-sample yaw errors in degrees.
    """
    rotation_gt_np = rotation_gt.cpu().numpy()
    rotation_pred_np = rotation_pred.cpu().numpy()
    yaw_errors = []
    for b in range(rotation_pred_np.shape[0]):
        R_pred = rotation_pred_np[b]
        R_gt = rotation_gt_np[b]

        yaw_pred = np.degrees(np.arctan2(R_pred[1, 0], R_pred[0, 0]))
        yaw_gt = np.degrees(np.arctan2(R_gt[1, 0], R_gt[0, 0]))

        diff = np.abs(yaw_pred - yaw_gt)
        err = float(np.min([diff, 360 - diff]))
        yaw_errors.append(err)
    return yaw_errors


def build_metrics_dict(lateral, longi, loc, yaw_err, solver, mode):
    """Assemble the summary metric dict of one solver from sample lists.

    Args:
        lateral, longi: per-sample lateral (x) / longitudinal (y) errors in
            meters.
        loc: per-sample euclidean location error (meters).
        yaw_err: per-sample yaw errors in degrees.
        solver: solver name used as metric-key tag; one of 'procrustes',
            'e2e', 'bisection', 'ransac'.
        mode: 'with' | 'nomask'; 'nomask' appends '_nomask' to every key.
    Returns:
        dict of metric name -> float, e.g. ``loc_e2e_mean``,
        ``yaw_ransac_R@1deg``, ``lateral_procrustes_R@1m_nomask``.
    """
    suffix = '_nomask' if mode == 'nomask' else ''
    lat, lon = np.asarray(lateral), np.asarray(longi)
    loc_arr, yaw_arr = np.asarray(loc), np.asarray(yaw_err)

    def _r(arr, thr):
        return float(np.mean(arr < thr)) if arr.size else float('nan')

    values = {
        f'loc_{solver}_mean': float(loc_arr.mean()),
        f'loc_{solver}_median': float(np.median(loc_arr)),
        f'lateral_{solver}_mean': float(lat.mean()),
        f'lateral_{solver}_median': float(np.median(lat)),
        f'lateral_{solver}_R@1m': _r(lat, 1.0),
        f'lateral_{solver}_R@5m': _r(lat, 5.0),
        f'long_{solver}_mean': float(lon.mean()),
        f'long_{solver}_median': float(np.median(lon)),
        f'long_{solver}_R@1m': _r(lon, 1.0),
        f'long_{solver}_R@5m': _r(lon, 5.0),
        f'yaw_{solver}_mean': float(yaw_arr.mean()),
        f'yaw_{solver}_median': float(np.median(yaw_arr)),
        f'yaw_{solver}_R@1deg': _r(yaw_arr, 1.0),
        f'yaw_{solver}_R@5deg': _r(yaw_arr, 5.0),
    }
    return {k + suffix: v for k, v in values.items()}
