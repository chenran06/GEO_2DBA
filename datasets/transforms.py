"""Shared image pre-processing used by the dataset classes.

Conventions:
    Images are normalised with ImageNet mean/std after a bicubic
    anti-aliased resize (``img2tensor``) for the pinhole datasets
    (KITTI / RobotCar); the panorama datasets (CVACT / VIGOR) go through
    torchvision ``ColorJitter`` + ``Resize`` + ``ToTensor`` pipelines.
    ``safe_collate`` drops ``None`` entries produced by unreadable samples.
"""
import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.nn import functional as F
from torch.utils.data import default_collate
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def img2tensor(img_pil, input_h, input_w, train_mode=False, color_jitter=False):
    """PIL image -> normalised tensor via bicubic anti-aliased resize.

    Args:
        img_pil:      input PIL image (RGB).
        input_h:      target height in pixels.
        input_w:      target width in pixels.
        train_mode:   when True (and ``color_jitter``), apply the mild
            photometric augmentation used by the RobotCar pipeline.
        color_jitter: enable the RobotCar-style ColorJitter augmentation
            (KITTI keeps this disabled to match its legacy pipeline).

    Returns:
        torch.Tensor [3, input_h, input_w], ImageNet-normalised.

    Raises:
        None. Expects an already-converted RGB PIL image.
    """
    if train_mode and color_jitter:
        aug = transforms.ColorJitter(
            brightness=0.2 / 4,
            contrast=0.2 / 4,
            saturation=0.1 / 4,
            hue=0.0,
        )
        img_pil = aug(img_pil)

    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    x = F.interpolate(x[None], size=(input_h, input_w), mode='bicubic',
                      align_corners=False, antialias=True)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return x[0]


def rotation_matrix_from_yaw(yaw):
    """[B]-less single yaw (radians) -> [2, 2] SO(2) matrix R(yaw)."""
    return torch.tensor(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=torch.float32,
    )


def rotate_high_quality(img_pil, angle, center=None, expand=True,
                        border_value=(0, 0, 0)):
    """Rotate a PIL image with LANCZOS4 resampling (higher quality than PIL).

    Args:
        img_pil:      input PIL image.
        angle:        rotation angle, counter-clockwise positive (PIL-style).
        center:       rotation center (x, y); defaults to the image center.
        expand:       grow the canvas so the whole source stays visible;
                      False keeps the original size.
        border_value: fill color for areas outside the source.

    Returns:
        Rotated PIL image.

    Raises:
        cv2.error: propagated from ``warpAffine`` on malformed inputs.
    """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv.getRotationMatrix2D(center, angle, scale=1.0)

    if expand:
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        out_size = (new_w, new_h)
    else:
        out_size = (w, h)

    rotated = cv.warpAffine(
        img_np, M, out_size,
        flags=cv.INTER_LANCZOS4,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return Image.fromarray(rotated)


def safe_collate(batch):
    """Collate that skips ``None`` entries (failed sample loads).

    Args:
        batch: list of dataset items, possibly containing ``None``.
    Returns:
        Default-collated batch, or ``None`` when every item failed.
    """
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return default_collate(batch)


def get_transform_grd(train, ground_image_size, satellite_image_size):
    """Torchvision pipelines for the panorama datasets (CVACT / VIGOR).

    Args:
        train:                True adds the mild ColorJitter augmentation.
        ground_image_size:    (H, W) resize target for ground panoramas.
        satellite_image_size: (H, W) resize target for satellite images.

    Returns:
        (transform_grd, transform_sat): Compose objects mapping PIL images
        to tensors.
    """
    if train:
        jitter = dict(brightness=0.2 / 4, contrast=0.2 / 4,
                      saturation=0.1 / 4, hue=0.0)
        transform_grd = transforms.Compose([
            transforms.ColorJitter(**jitter),
            transforms.Resize(ground_image_size),
            transforms.ToTensor(),
        ])
        transform_sat = transforms.Compose([
            transforms.ColorJitter(**jitter),
            transforms.Resize(satellite_image_size),
            transforms.ToTensor(),
        ])
    else:
        transform_grd = transforms.Compose([
            transforms.Resize(ground_image_size),
            transforms.ToTensor(),
        ])
        transform_sat = transforms.Compose([
            transforms.Resize(satellite_image_size),
            transforms.ToTensor(),
        ])
    return transform_grd, transform_sat
