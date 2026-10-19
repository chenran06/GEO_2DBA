"""RobotCar gravity-leveling geometry shared by depth inference and mask extraction.

The RobotCar ground images are levelled with a per-frame roll/pitch rotation
(``stereo_preprocess_v2`` stage).  That rotation maps some output pixels
outside the de-distorted image; ``remap`` fills them black.  Both inference
scripts reconstruct this black border geometrically (no re-detection) from
the per-run intrinsics and the roll/pitch values logged in
``process_log.csv`` — verified to match the actual border pixel-by-pixel.
"""

import csv
import os

import numpy as np

IMG_H, IMG_W = 960, 1280
MODEL_NAME = 'stereo_narrow_left'


def load_leveling_ctx(run_dir):
    """Load one run's leveling context: K, R_sensor_cam and per-frame roll/pitch (rad).

    Same source as the leveling stage itself: intrinsics/extrinsics from
    ``<run>/stereo_preprocess_v2/<MODEL_NAME>.txt`` and
    ``R_c = R_sc^T * Ry(pitch) * Rx(roll) * R_sc``.  roll/pitch are read from
    ``process_log.csv`` (degrees, %.4f ≈ 0.002 px — sufficient).
    """
    with open(os.path.join(run_dir, 'stereo_preprocess_v2', MODEL_NAME + '.txt')) as f:
        vals = [float(x) for x in f.readline().split()]
        G = np.array([[float(x) for x in f.readline().split()] for _ in range(4)])
    K = np.array([[vals[0], 0, vals[2]], [0, vals[1], vals[3]], [0, 0, 1]])
    log = {}
    with open(os.path.join(run_dir, 'stereo_preprocess_v2', 'process_log.csv')) as f:
        for r in csv.DictReader(f):
            log[r['filename']] = (np.radians(float(r['roll_deg'])),
                                  np.radians(float(r['pitch_deg'])))
    return K, G[:3, :3], log


def black_border_mask(roll, pitch, K, R_sc, K_inv, grid):
    """Reconstruct the leveling black border: H^-1 lands outside the image.

    Pixel-wise identical to the ``inside`` test of the leveling remap.
    ``grid`` is the (3, H*W) stack of output pixel homogeneous coordinates
    (build once per process with :func:`pixel_grid`).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_c = R_sc.T @ Ry @ Rx @ R_sc
    H_inv = K @ R_c.T @ K_inv
    uv = H_inv @ grid
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    return ((u < 0) | (u > IMG_W - 1) | (v < 0) | (v > IMG_H - 1)).reshape(IMG_H, IMG_W)


def pixel_grid():
    """Homogeneous pixel grid (3, H*W) for :func:`black_border_mask`."""
    yy, xx = np.meshgrid(np.arange(IMG_H, dtype=np.float64),
                         np.arange(IMG_W, dtype=np.float64), indexing='ij')
    return np.vstack([xx.ravel(), yy.ravel(), np.ones(IMG_H * IMG_W)])
