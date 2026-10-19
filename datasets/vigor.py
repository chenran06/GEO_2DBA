"""VIGOR panorama satellite-ground dataset.

Depth comes from pre-computed UniK3D metric depth maps (16-bit PNG, mm),
horizontally rolled together with the panorama for random-orientation
training. The ``samearea`` split draws train/val from a random 80/20
subset of the four-city balanced train list; ``crossarea`` trains on
NewYork+Seattle and evaluates on SanFrancisco+Chicago.

Returned tuple (9 items):
    grd_panorama, depth, sat, gt_loc, rotation_gt, city, resolution,
    yaw, mpp
"""
import logging
import os
from pathlib import Path

import ast
import configparser
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.transforms import get_transform_grd, safe_collate

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.ini'
_config = configparser.ConfigParser()
_config.read(_CONFIG_PATH)

MAX_DEPTH_METERS = 35.0
CITY_RESOLUTIONS = {
    'NewYork': 0.113248,
    'Seattle': 0.100817,
    'SanFrancisco': 0.118141,
    'Chicago': 0.111262,
}
# Satellite meters/pixel at the VIGOR reference latitude (lat-corrected per
# sample via cos(lat)).
K_MPP = 0.14929107086948487

# samearea: fraction of the balanced train list used for training.
TRAIN_SPLIT = 0.8
NUM_WORKERS = 4

GROUND_IMAGE_SIZE = ast.literal_eval(_config.get('VIGOR', 'ground_image_size'))
SATELLITE_IMAGE_SIZE = ast.literal_eval(_config.get('VIGOR', 'satellite_image_size'))


class VIGORDataset(Dataset):
    """VIGOR ground panorama <-> satellite tile pairs.

    Args:
        root:               dataset root (contains <city>/panorama,
                            <city>/satellite and the label lists).
        label_root:         label directory name under root.
        split:              'samearea' | 'crossarea'.
        train:              selects the label file and the jitter pipeline.
        random_orientation: panorama roll amplitude in degrees (0 = fixed).
        transform:          optional (grd, sat) transform override.
    """

    def __init__(self, root, label_root='splits',
                 split='samearea', train=True, random_orientation=0,
                 transform=None):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.random_orientation = random_orientation
        self.grdimage_transform, self.satimage_transform = (
            transform if transform is not None
            else get_transform_grd(self.train, GROUND_IMAGE_SIZE,
                                   SATELLITE_IMAGE_SIZE))

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta = self._load_ground_data()
        self.data_size = len(self.grd_list)
        logger.info('Loaded %d VIGOR entries (split=%s, train=%s)',
                    self.data_size, split, train)

    def _get_city_list(self):
        """Cities used by this split/half.

        Returns:
            list of city names.

        Raises:
            ValueError: on unsupported split.
        """
        if self.split == 'samearea':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'crossarea':
            return ['NewYork', 'Seattle'] if self.train \
                else ['SanFrancisco', 'Chicago']
        raise ValueError(f'Unsupported split: {self.split}')

    def _load_satellite_data(self):
        """Read every city's ``satellite_list.txt``.

        Returns:
            (sat_list, sat_index_dict): path array and filename -> row index.
        """
        sat_list, sat_index_dict = [], {}
        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(
                self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(
                        self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            logger.info('Loaded %s, %d entries', sat_list_fname, idx)
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        """Read the ground-panorama label file of every city.

        Returns:
            (grd_list, label, delta): panorama paths, positive satellite
            indices [N, 4] and pixel offsets [N, 4, 2].
        """
        grd_list, label_list, delta_list = [], [], []
        idx = 0
        for city in self.city_list:
            label_fname = self._get_label_file(city)
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array(
                        [self.sat_index_dict[data[i]]
                         for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array(
                        [data[2:4], data[5:7], data[8:10], data[11:13]]
                    ).astype(np.float32)
                    grd_list.append(
                        os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    idx += 1
            logger.info('Loaded %s, %d entries', label_fname, idx)
        return grd_list, np.array(label_list), np.array(delta_list)

    def _get_label_file(self, city):
        """Label file path for a city (samearea reuses the balanced train
        list for both halves; crossarea uses pano_label_balanced.txt)."""
        if self.split == 'samearea':
            return os.path.join(self.root, self.label_root, city,
                                'same_area_balanced_train.txt')
        return os.path.join(self.root, self.label_root, city,
                            'pano_label_balanced.txt')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        try:
            grd = self._load_image(self.grd_list[idx])
            if grd is None:
                raise ValueError('Ground image is None')
            grd = self.grdimage_transform(grd)

            depth_path = os.path.splitext(
                self.grd_list[idx].replace('panorama', 'unik3d_depth'))[0] + '.png'
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

            sat, row_offset, col_offset, raw_height = \
                self._load_satellite_image(idx)
            if sat is None:
                raise ValueError('Satellite image is None')

            # GT location: satellite-pixel offsets, flipped to grd->sat.
            gt_loc = -torch.tensor([[-row_offset, col_offset]])
            yaw = -yaw
            rotation_matrix = torch.tensor(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
                dtype=torch.float32)

            sat_path = self.sat_list[self.label[idx][0]]
            sat_filename = sat_path.split('/')[-1]
            lat = float(sat_filename.split('_')[1])
            mpp = K_MPP * np.cos(np.radians(lat))

            city, resolution = self._get_city_info(self.grd_list[idx],
                                                   raw_height)
            return grd_rolled, depth_rolled, sat, gt_loc, rotation_matrix, \
                city, resolution, yaw, mpp

        except Exception as e:
            logger.warning('Skipping VIGOR sample %d due to error: %s (%s)',
                           idx, e, self.grd_list[idx])
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

    def _load_satellite_image(self, idx):
        """Load the positive satellite tile and rescale its GT offsets.

        Args:
            idx: sample index.
        Returns:
            (sat_pil, row_offset, col_offset, raw_height) in resized-satellite
            pixel units; ``None``-filled on failure.
        """
        try:
            pos_index = 0
            row_offset, col_offset = self.delta[idx, pos_index]
            sat = self._load_image(self.sat_list[self.label[idx][pos_index]])
            raw_width, raw_height = sat.size
            sat = self.satimage_transform(sat)

            row_offset = row_offset / raw_height * sat.size(1)
            col_offset = col_offset / raw_width * sat.size(2)
            return sat, row_offset, col_offset, raw_height
        except Exception as e:
            logger.warning('Failed to load VIGOR satellite at %d: %s', idx, e)
            return None, None, None, None

    def _get_city_info(self, path, raw_height):
        """City name and lat-corrected meters/pixel of a satellite tile.

        Args:
            path:       ground panorama path (contains the city name).
            raw_height: raw satellite tile height in pixels.
        Returns:
            (city, resolution) where resolution is m/pixel after resize to
            SATELLITE_IMAGE_SIZE[0]; ('Unknown', 1.0) when unmatched.
        """
        for city, resolution in CITY_RESOLUTIONS.items():
            if city in path:
                return city, resolution * raw_height / SATELLITE_IMAGE_SIZE[0]
        return 'Unknown', 1.0


def build_dataloaders(dataset_root, split, batch_size, random_orientation,
                      num_workers=NUM_WORKERS):
    """Build VIGOR train / val loaders for one split.

    samearea draws a random 80/20 Subset split of the balanced train list
    (the shuffle consumes the global numpy RNG); crossarea uses the full
    train half for training and the held-out cities for evaluation.

    Args:
        dataset_root:       VIGOR dataset root.
        split:              'samearea' | 'crossarea'.
        batch_size:         batch size for both loaders.
        random_orientation: panorama roll amplitude in degrees.
        num_workers:        DataLoader workers.
    Returns:
        (train_loader, val_loader), both using :func:`safe_collate`.
    """
    dataset = VIGORDataset(root=dataset_root, split=split, train=True,
                           random_orientation=random_orientation)
    dataset_val = VIGORDataset(root=dataset_root, split=split, train=False,
                               random_orientation=random_orientation)
    logger.info('Total dataset length: %d, Validation set size: %d',
                len(dataset), len(dataset_val))

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_idx = int(len(dataset) * TRAIN_SPLIT)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    if split == 'samearea':
        training_set = Subset(dataset, train_indices)
        val_set = Subset(dataset_val, val_indices)
    else:
        training_set = dataset
        val_set = dataset_val

    train_dataloader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=safe_collate)
    val_dataloader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=safe_collate)
    return train_dataloader, val_dataloader
