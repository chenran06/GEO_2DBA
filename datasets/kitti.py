"""KITTI satellite-ground dataset (pinhole camera pipeline).

Each sample pairs a KITTI left-camera frame with its pre-rendered satellite
tile. Ground-truth shifts/orientations are drawn randomly at training time
and read from the split-file line at evaluation time (``name dx dy theta``).

Returned tuple (11 items, no ``mask_sat``):
    sat_map, grd_img, depth, camera_k, gt_loc, rotation_gt,
    mask_grd, [grd_w, grd_h], yaw, (left_img_name, satmap_name, file_name),
    mpp
where ``mpp`` is the sample's satellite meters-per-pixel computed from the
oxts latitude so metric BEV grids can be built per sample.
"""
import logging
import os

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import img2tensor, rotate_high_quality

logger = logging.getLogger(__name__)

Default_lat = 49.015
Satmap_zoom = 18

GrdImg_H = 416
GrdImg_W = 1408
GrdOriImg_H = 375
GrdOriImg_W = 1242
SatMap_original_sidelength = 656
SatMap_process_sidelength = 656

satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

MAX_DEPTH_METERS = 40.0


def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=1):
    """Satellite meters-per-pixel at ``lat``/``zoom`` for the Web-Mercator tiles.

    Args:
        lat:   latitude in degrees (KITTI area default 49.015).
        zoom:  satellite tile zoom level (18).
        scale: extra downscale factor applied to the tiles.
    Returns:
        float meters per pixel.
    """
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / (2 ** zoom)
    meter_per_pixel /= 2
    meter_per_pixel /= scale
    return meter_per_pixel


def _load_metric_depth_file(path, max_depth=MAX_DEPTH_METERS):
    """Load a 16-bit PNG metric depth map (1/256 m per unit) and clip it.

    Args:
        path:      depth PNG path.
        max_depth: clip upper bound in meters.
    Returns:
        np.ndarray float32 [H, W] in meters, or ``None`` when unreadable.
    """
    try:
        metric_depth = cv.imread(path, cv.IMREAD_UNCHANGED)
        if metric_depth is None:
            raise ValueError('cv2.imread returned None')
        metric_depth = metric_depth.astype(np.float32) / 256.0
        return np.clip(metric_depth, 0, max_depth)
    except Exception as e:
        logger.warning('Unreadable depth image: %s (%s)', path, e)
        return None


def _read_left_camera_k(root, day_dir):
    """Read the left-camera ``P_rect_02`` calibration and rescale it to the
    processed ground resolution.

    Args:
        root:    dataset root directory.
        day_dir: capture day folder name (first 10 chars of the sample name).
    Returns:
        torch.Tensor [3, 3] float32 intrinsics for GrdImg_H x GrdImg_W.
    Raises:
        ValueError: if the calibration file is missing or has no P_rect_02.
    """
    calib_file_name = os.path.join(root, grdimage_dir, day_dir,
                                   'calib_cam_to_cam.txt')
    with open(calib_file_name, 'r') as f:
        for line in f:
            if 'P_rect_02' not in line:
                continue
            values = line.split(':')[1].strip().split(' ')
            fx = float(values[0]) * GrdImg_W / GrdOriImg_W
            cx = float(values[2]) * GrdImg_W / GrdOriImg_W
            fy = float(values[5]) * GrdImg_H / GrdOriImg_H
            cy = float(values[6]) * GrdImg_H / GrdOriImg_H
            return torch.from_numpy(np.asarray(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
    raise ValueError(f'P_rect_02 not found in {calib_file_name}')


class SatGrdDataset(Dataset):
    """KITTI ground-satellite pairs with random shift/rotation augmentation.

    Args:
        root:           dataset root (contains ``satmap/`` and ``raw_data/``).
        file:           split file path; test lines append
                        ``gt_shift_x gt_shift_y theta`` to the sample name.
        shift_range_lat / shift_range_lon: random GT shift range in meters.
        rotation_range: random satellite rotation amplitude in degrees.
        train:          training mode (random GT vs split-file GT, and the
                        photometric augmentation is inactive either way —
                        kept for interface parity with the other datasets).
    """

    def __init__(self, root, file, shift_range_lat=20, shift_range_lon=20,
                 rotation_range=10, train=True):
        self.root = root
        self.train = train
        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel
        self.rotation_range = rotation_range

        with open(file, 'r') as f:
            self.file_name = [line.rstrip('\n') for line in f]
        logger.info('Loaded %d KITTI samples from %s (train=%s)',
                    len(self.file_name), file, train)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        if self.train:
            file_name = self.file_name[idx]
            gt_shift_x = np.random.uniform(-1, 1)  # right positive, along heading
            gt_shift_y = np.random.uniform(-1, 1)  # up positive, across heading
            random_ori = np.random.uniform(-1, 1) * self.rotation_range
        else:
            file_name, gt_shift_x, gt_shift_y, theta = \
                self.file_name[idx].split(' ')
            gt_shift_x = float(gt_shift_x)
            gt_shift_y = float(gt_shift_y)
            random_ori = float(theta) * self.rotation_range

        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        left_camera_k = _read_left_camera_k(self.root, day_dir)

        satmap_name = os.path.join(self.root, satmap_dir, file_name)
        with Image.open(satmap_name, 'r') as sat_map:
            sat_map = sat_map.convert('RGB')

        oxts_file_name = os.path.join(
            self.root, grdimage_dir, drive_dir, oxts_dir,
            image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            oxts_fields = f.readline().split(' ')
            heading = float(oxts_fields[5])
        # Per-sample satellite scale from this frame's own latitude.
        mpp = get_meter_per_pixel(lat=float(oxts_fields[0]))

        left_img_name = os.path.join(
            self.root, grdimage_dir, drive_dir, left_color_camera_dir,
            image_no.lower())

        grd_img_left = Image.open(left_img_name).convert('RGB')
        w, h = grd_img_left.size

        metric_depth_name = left_img_name.replace(
            'raw_data', 'depth_anythingv2_depth').replace('image_02/data', 'depth')
        depth = _load_metric_depth_file(metric_depth_name)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W),
                              mode='nearest').squeeze(0)

        mask_grd_name = left_img_name.replace('raw_data', 'mask2former')
        mask_grd = cv.imread(mask_grd_name)
        mask_grd = np.where(mask_grd == 255, 0, 1).astype(np.uint8)
        mask_grd = cv.resize(mask_grd, (GrdImg_W // 16, GrdImg_H // 16),
                             interpolation=cv.INTER_NEAREST)

        # Align the satellite north-up image to the camera heading.
        sat_rot = rotate_high_quality(sat_map, (-heading) / np.pi * 180)

        dx = CameraGPS_shift_left[0] / self.meter_per_pixel
        dy = CameraGPS_shift_left[1] / self.meter_per_pixel
        dx += gt_shift_x * self.shift_range_pixels_lon
        dy += -gt_shift_y * self.shift_range_pixels_lat

        sat_rand_shift = sat_rot.transform(
            sat_rot.size, Image.AFFINE,
            (1, 0, int(dx), 0, 1, int(dy)),
            resample=Image.BILINEAR)

        sat_rand_shift_rand_rot = rotate_high_quality(sat_rand_shift, random_ori)
        sat_map = TF.center_crop(sat_rand_shift_rand_rot,
                                 SatMap_original_sidelength)

        sat_map = img2tensor(sat_map, SatMap_process_sidelength,
                             SatMap_process_sidelength)
        grd_img_left = img2tensor(grd_img_left, GrdImg_H, GrdImg_W)

        # GT location: rotated-frame pixel offsets -> satellite pixel offsets.
        x_offset = int(gt_shift_x * self.shift_range_pixels_lon *
                       np.cos(random_ori / 180 * np.pi) -
                       gt_shift_y * self.shift_range_pixels_lat *
                       np.sin(random_ori / 180 * np.pi))
        y_offset = int(-gt_shift_y * self.shift_range_pixels_lat *
                       np.cos(random_ori / 180 * np.pi) -
                       gt_shift_x * self.shift_range_pixels_lon *
                       np.sin(random_ori / 180 * np.pi))
        x_offset = x_offset / SatMap_original_sidelength * SatMap_process_sidelength
        y_offset = y_offset / SatMap_original_sidelength * SatMap_process_sidelength

        yaw = random_ori * (np.pi / 180)
        gt_loc = -torch.tensor([[y_offset, x_offset]])

        rotation_gt = torch.tensor(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=torch.float32)

        return (sat_map, grd_img_left, depth, left_camera_k, gt_loc,
                rotation_gt, mask_grd, [w, h], yaw,
                (left_img_name, satmap_name, file_name), mpp)


def build_dataloaders(root, splits_dir, rotation_range, batch_size,
                      num_workers=4, shuffle=True):
    """Build the KITTI train / test1 (cross-day) / test2 (same-day) loaders.

    Args:
        root:           dataset root directory.
        splits_dir:     directory holding ``train/test1/test2_files.txt``.
        rotation_range: satellite rotation amplitude in degrees (10 or 180).
        batch_size:     batch size for all three loaders.
        num_workers:    DataLoader workers.
        shuffle:        whether to shuffle the training loader.
    Returns:
        (train_loader, test1_loader, test2_loader).

    Raises:
        FileNotFoundError: if any split file is missing.
    """
    loaders = []
    for split in ('train_files.txt', 'test1_files.txt', 'test2_files.txt'):
        split_path = os.path.join(splits_dir, split)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f'Missing KITTI split file: {split_path}')
        train = split.startswith('train')
        dataset = SatGrdDataset(
            root=root,
            file=split_path,
            shift_range_lat=20,
            shift_range_lon=20,
            rotation_range=rotation_range,
            train=train,
        )
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if train else False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
        ))
    return tuple(loaders)
