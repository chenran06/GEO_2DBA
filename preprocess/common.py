"""Shared helpers for the pre-processing scripts.

The external model repositories (Depth-Anything-V2, UniK3D) are NOT vendored:
clone them under ``<project_root>/external/`` as described in
``preprocess/README.md`` and pass ``--external-root`` (default
``<project_root>/external``) to the scripts.
"""

import configparser
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.ini"

DAV2_CHECKPOINT = "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"


def dataset_root_from_config(section):
    """Return the ``dataset_root`` of ``section`` from config.ini."""
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f"config.ini not found at {CONFIG_PATH}")
    return config[section]["dataset_root"]


def save_depth_png(depth, out_path, scale=256.0):
    """Save metric depth as 16-bit PNG (``round(depth * scale)`` uint16).

    ``scale=256`` is the Depth-Anything-V2 convention read back by
    ``datasets/kitti.py`` / ``datasets/robotcar.py``; ``scale=1000`` (mm) is
    the UniK3D convention read back by ``datasets/vigor.py`` /
    ``datasets/cvact.py``.
    """
    metric = np.round(depth * scale).astype(np.uint16)
    cv2.imwrite(str(out_path), metric)


def visualize_pair(raw_image, depth, out_path):
    """Save a side-by-side raw image / colourised depth visualisation."""
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("Spectral")
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        vis = (depth - dmin) / (dmax - dmin) * 255.0
    else:
        vis = np.zeros_like(depth)
    vis = (cmap(vis.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    h = min(raw_image.shape[0], vis.shape[0])
    vis = cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h))
    img = raw_image if raw_image.shape[0] == h else \
        cv2.resize(raw_image, (int(raw_image.shape[1] * h / raw_image.shape[0]), h))
    cv2.imwrite(str(out_path), cv2.hconcat([img, np.full((h, 30, 3), 255, np.uint8), vis]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
