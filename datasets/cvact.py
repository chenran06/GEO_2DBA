"""CVACT panorama satellite-ground dataset.

Depth comes from pre-computed UniK3D metric depth maps (16-bit PNG, mm),
horizontally rolled together with the panorama for random-orientation
training. Unreadable samples yield ``None`` and are dropped by
:func:`datasets.transforms.safe_collate`.

Returned tuple (7 items):
    grd_panorama, depth, sat, gt_loc, rotation_gt, yaw, mpp
"""
import logging
import os
from pathlib import Path

import ast
import configparser
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.cvact_read import load_cvact
from datasets.transforms import get_transform_grd, safe_collate

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.ini'
_config = configparser.ConfigParser()
_config.read(_CONFIG_PATH)

MAX_DEPTH_METERS = 35.0

# CVACT satellite tiles: 0.128 m/pixel on the raw 1200x1200 grid. The square
# (656, 656) center crop keeps that scale (resize is then an identity), so the
# per-sample metric extent is K_MPP * 656 = 83.968 m; the training script
# builds it per sample as grid_size_h = mmp * sat_size.
K_MPP = 0.128

GROUND_IMAGE_SIZE = ast.literal_eval(_config.get('CVACT', 'ground_image_size'))
SATELLITE_IMAGE_SIZE = ast.literal_eval(_config.get('CVACT', 'satellite_image_size'))


class CVACTdataset(Dataset):
    """CVACT ground panorama <-> satellite tile pairs.

    Args:
        dataset:            list of (grd_path, sat_path, lat, lon) entries
                            from :func:`datasets.cvact_read.load_cvact`.
        root:               dataset root (unused for lookup, kept for parity).
        train:              enables the ColorJitter ground/satellite pipelines.
        random_orientation: panorama roll amplitude in degrees (0 = fixed).
        transform:          optional (grd, sat) transform override.
    """

    def __init__(self, dataset, root, train=True,
                 random_orientation=0, transform=None):
        self.root = root
        self.train = train
        self.random_orientation = random_orientation
        self.grdimage_transform, self.satimage_transform = (
            transform if transform is not None
            else get_transform_grd(self.train, GROUND_IMAGE_SIZE,
                                   SATELLITE_IMAGE_SIZE))
        shift_range = 10.0  # meters
        self.dataset = dataset
        self.shift_range_pixels_lat = shift_range / K_MPP
        self.shift_range_pixels_lon = shift_range / K_MPP
        logger.info('Loaded %d CVACT entries (train=%s)',
                    len(self.dataset), train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            grd_path, sat_path, lat, lon = self.dataset[idx]
            grd = self._load_image(grd_path)
            if grd is None:
                raise ValueError('Ground image is None')
            grd = self.grdimage_transform(grd)

            depth_path = os.path.splitext(
                grd_path.replace('streetview', 'unik3d_depth'))[0] + '.png'
            depth = self._load_metric_depth(depth_path)
            if depth is None:
                raise ValueError('Depth image is None')
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth = F.interpolate(depth, size=GROUND_IMAGE_SIZE,
                                  mode='nearest').squeeze(0)

            # Random horizontal roll of panorama + depth (orientation aug).
            rotation = np.random.uniform(-self.random_orientation / 360,
                                         self.random_orientation / 360)
            grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
            depth_rolled = torch.roll(depth, int(round(rotation * depth.size(2))), dims=2)

            yaw = -rotation * 360 * (np.pi / 180)

            sat = self._load_image(sat_path)
            if sat is None:
                raise ValueError('Satellite image is None')

            gt_shift_x = np.random.uniform(-1, 1)  # right positive, along heading
            gt_shift_y = np.random.uniform(-1, 1)  # up positive, across heading
            dx = int(gt_shift_x * self.shift_range_pixels_lon)
            dy = int(-gt_shift_y * self.shift_range_pixels_lat)

            sat_rand_shift = sat.transform(
                sat.size, Image.AFFINE,
                (1, 0, dx, 0, 1, dy),
                resample=Image.BILINEAR)
            # Square crop keeps the 0.128 m/px tile scale (the legacy
            # GROUND_IMAGE_SIZE crop here anisotropically squashed the tile).
            sat_map = TF.center_crop(sat_rand_shift, SATELLITE_IMAGE_SIZE)
            sat = self.satimage_transform(sat_map)

            gt_loc = -torch.tensor([[dy, dx]])
            yaw = -yaw
            rotation_matrix = torch.tensor(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
                dtype=torch.float32)

            return grd_rolled, depth_rolled, sat, gt_loc, rotation_matrix, \
                yaw, K_MPP

        except Exception as e:
            logger.warning('Skipping CVACT sample %d due to error: %s', idx, e)
            return None

    def _load_image(self, path):
        """Load a PIL RGB image; ``None`` when unreadable."""
        try:
            with Image.open(path) as image:
                return image.convert('RGB')
        except Exception as e:
            logger.warning('Unreadable image: %s (%s)', path, e)
            return None

    def _load_metric_depth(self, path, max_depth=MAX_DEPTH_METERS):
        """Load a UniK3D depth map (PNG mm or npy m), clipped to max_depth.

        Returns ``None`` for unreadable files or all-invalid depth.
        """
        try:
            if path.endswith('.npy'):
                depth = np.load(path)
            else:
                with Image.open(path) as depth_image:
                    depth = np.array(depth_image).astype(np.float32) / 1000.0
            depth = np.clip(depth, 0, max_depth)
            if np.all(depth == max_depth):
                logger.warning('all depth are larger than defined max depth')
                return None
            return depth
        except Exception as e:
            logger.warning('Unreadable depth image: %s (%s)', path, e)
            return None


def build_dataloaders(dataset_root, batch_size, random_orientation,
                      num_workers=4):
    """Build CVACT train / val loaders (val split stands in as test).

    Args:
        dataset_root:       CVACT root containing ACT_data.mat.
        batch_size:         batch size for both loaders.
        random_orientation: panorama roll amplitude in degrees.
        num_workers:        DataLoader workers.
    Returns:
        (train_loader, val_loader), both using :func:`safe_collate`.
    """
    train_set, test_set = load_cvact(dataset_root)
    logger.info('CVACT splits: train %d, test %d',
                len(train_set), len(test_set))

    dataset = CVACTdataset(train_set, root=dataset_root, train=True,
                           random_orientation=random_orientation)
    dataset_val = CVACTdataset(test_set, root=dataset_root, train=False,
                               random_orientation=random_orientation)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=safe_collate)
    val_dataloader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=safe_collate)
    return train_dataloader, val_dataloader
