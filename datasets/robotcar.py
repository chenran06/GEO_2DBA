"""RobotCar satellite-ground dataset (pinhole camera pipeline).

Samples come from a CSV with columns
``(grd_path, sat_path, depth_path, lat, lon, yaw)``; the 80/20 train/test
split is a fixed ``random_state=42`` sklearn split. Ground-truth shifts and
satellite rotations are drawn randomly in both halves (the evaluation split
therefore keeps the same random-shift protocol as training).

Returned tuple (13 items, no ``mask_sat``):
    sat_map, grd_img, depth, camera_k, gt_loc, rotation_gt,
    mask_grd, [grd_w, grd_h], yaw, (grd_path, sat_path, depth_path),
    mpp, lat, lon
"""
import logging

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import img2tensor, rotate_high_quality

logger = logging.getLogger(__name__)

camera_k = [[964.828979, 0.0, 643.788025],
            [0.0, 964.828979, 484.407990],
            [0.0, 0.0, 1.0]]

# Cityscapes dynamic-class ids removed from the ground validity mask.
DYNAMIC_CLASSES = [8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 255]

GrdImg_H = 480
GrdImg_W = 640
SatMap_original_sidelength = 1024
SatMap_process_sidelength = 656
MMP = 0.149291070869


class robotcarSatGrdDataset(Dataset):
    """RobotCar ground-satellite pairs with random shift/rotation augmentation.

    Args:
        dataset:        list of CSV rows (grd, sat, depth, lat, lon, yaw).
        shift_range:    random GT shift range in meters.
        rotation_range: random satellite rotation amplitude in degrees.
        train:          enables the ColorJitter photometric augmentation.
    """

    def __init__(self, dataset, shift_range=10, rotation_range=10, train=True):
        self.dataset = dataset
        self.train = train
        self.shift_range = shift_range
        self.rotation_range = rotation_range
        logger.info('Loaded %d RobotCar samples (train=%s).',
                    len(self.dataset), train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        grd_path, sat_path, depth_path, lat, lon, yaw = self.dataset[idx]
        lat, lon, yaw = float(lat), float(lon), float(yaw)
        gt_shift_x = np.random.uniform(-1, 1)  # right positive, along heading
        gt_shift_y = np.random.uniform(-1, 1)  # up positive, across heading
        random_ori = np.random.uniform(-1, 1) * self.rotation_range

        with Image.open(sat_path, 'r') as sat_map_in:
            sat_map_pil = sat_map_in.convert('RGB')

        grd_img_left = Image.open(grd_path).convert('RGB')
        w, h = grd_img_left.size

        depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)  # uint16
        depth = depth.astype(np.float32) / 256.0            # restore meters
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W),
                              mode='nearest').squeeze(0)

        mask_grd_name = depth_path.replace('depth', 'mask2former')
        mask_grd = cv.imread(mask_grd_name)
        mask_dynamic = np.isin(mask_grd, DYNAMIC_CLASSES)
        mask_static = np.where(mask_dynamic, 0, 1).astype(np.uint8)
        mask_grd = cv.resize(mask_static, (GrdImg_W // 16, GrdImg_H // 16),
                             interpolation=cv.INTER_NEAREST)

        # Rotate north-up satellite to vehicle heading (east = heading).
        sat_rot = rotate_high_quality(sat_map_pil, (yaw / np.pi * 180) - 90)

        dx = 0
        dy = 0
        mmp = MMP * np.cos(lat * np.pi / 180)

        dx += gt_shift_x * self.shift_range / mmp
        dy += -gt_shift_y * self.shift_range / mmp

        sat_rand_shift = sat_rot.transform(
            sat_rot.size, Image.AFFINE,
            (1, 0, int(dx), 0, 1, int(dy)),
            resample=Image.BILINEAR)

        sat_rand_shift_rand_rot = rotate_high_quality(sat_rand_shift, random_ori)
        sat_map = TF.center_crop(sat_rand_shift_rand_rot,
                                 SatMap_original_sidelength)

        sat_map = img2tensor(sat_map, SatMap_process_sidelength,
                             SatMap_process_sidelength,
                             train_mode=self.train, color_jitter=True)
        grd_img_left = img2tensor(grd_img_left, GrdImg_H, GrdImg_W,
                                  train_mode=self.train, color_jitter=True)
        camera_k_i = torch.tensor(camera_k) / 2

        # GT location: rotated-frame pixel offsets -> satellite pixel offsets.
        x_offset = int(dx * np.cos(random_ori / 180 * np.pi) +
                       dy * np.sin(random_ori / 180 * np.pi))
        y_offset = int(dy * np.cos(random_ori / 180 * np.pi) -
                       dx * np.sin(random_ori / 180 * np.pi))
        x_offset = x_offset / SatMap_original_sidelength * SatMap_process_sidelength
        y_offset = y_offset / SatMap_original_sidelength * SatMap_process_sidelength
        mmp = mmp * SatMap_original_sidelength / SatMap_process_sidelength

        yaw = random_ori * (np.pi / 180)
        gt_loc = -torch.tensor([[y_offset, x_offset]])

        rotation_gt = torch.tensor(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=torch.float32)

        return (sat_map, grd_img_left, depth, camera_k_i, gt_loc,
                rotation_gt, mask_grd, [w, h], yaw,
                (grd_path, sat_path, depth_path), mmp, lat, lon)


def build_dataloaders(csv_path, rotation_range, batch_size, shift_range=10,
                      num_workers=4, shuffle=True):
    """Build RobotCar train / test loaders from the dataset CSV.

    The 80/20 split is deterministic (``random_state=42``), matching the
    legacy pipeline.

    Args:
        csv_path:       CSV file with columns
                        (grd_path, sat_path, depth_path, lat, lon, yaw).
        rotation_range: satellite rotation amplitude in degrees (10 or 180).
        batch_size:     batch size for both loaders.
        shift_range:    random GT shift range in meters.
        num_workers:    DataLoader workers.
        shuffle:        whether to shuffle the training loader.
    Returns:
        (train_loader, test1_loader).
    """
    dataset_list = pd.read_csv(csv_path).values.tolist()
    train_list, test_list = train_test_split(
        dataset_list, test_size=0.2, random_state=42)

    train_set = robotcarSatGrdDataset(
        dataset=train_list, shift_range=shift_range,
        rotation_range=rotation_range, train=True)
    test1_set = robotcarSatGrdDataset(
        dataset=test_list, shift_range=shift_range,
        rotation_range=rotation_range, train=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=num_workers, drop_last=False)
    test1_loader = DataLoader(
        test1_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers, drop_last=False)
    return train_loader, test1_loader
