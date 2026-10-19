"""Metric BEV grids and ground feature-map sampling grids."""
import numpy as np
import torch


def create_metric_grid(grid_size, resolution, batch_size, device):
    """Metric coordinates of a square BEV grid centred at the origin.

    The grid spans ``[-grid_size/2, grid_size/2]`` on both axes with
    ``resolution`` samples per axis (row-major over x then y).

    Args:
        grid_size: physical extent of the grid in meters (satellite tile size).
        resolution: samples per axis (``sat_bev_res``, e.g. 41).
        batch_size: replicate the grid for this many batch elements.
        device: target torch device.
    Returns:
        [B, resolution^2, 2] float32 metric (x, y) coordinates.
    """
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing="ij")
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_metric_grid_batch(grid_size, resolution, device):
    """Per-sample metric grids, one extent per batch element.

    Each sample's grid spans ``[-grid_size[b]/2, grid_size[b]/2]`` so that
    batches with heterogeneous latitudes / tile resolutions / resize factors
    each carry physically correct metric coordinates.

    Args:
        grid_size: [B] tensor (or list) of per-sample grid extents in meters,
            e.g. ``mpp * sat_size`` after the loader's resize chain.
        resolution: samples per axis (``sat_bev_res``, e.g. 41).
        device: target torch device.
    Returns:
        [B, resolution^2, 2] float32 metric (x, y) coordinates.
    """
    all_batch_coords = []
    for sz in grid_size:
        axis = torch.linspace(-sz / 2, sz / 2, resolution, device=device)
        metric_x, metric_y = torch.meshgrid(axis, axis, indexing="ij")
        coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
        all_batch_coords.append(coord)
    return torch.stack(all_batch_coords, dim=0)


def create_metric_grid_xy(grid_size, resolution, batch_size, device):
    """Same as :func:`create_metric_grid` but with xy (column-major) ordering.

    Used where grid ordering must match image (u, v) conventions.
    """
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing="xy")
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_image_grids(batch_size, device, grd_img_h, grd_img_w):
    """Pixel-coordinate grids of the ground feature map.

    Args:
        batch_size: replicate the grids for this many batch elements.
        device: target torch device.
        grd_img_h, grd_img_w: ground image size (loader-specific).
    Returns:
        (u_grid, v_grid), each [B, 1, H/16, W/16]: pixel coordinates of every
        ground feature cell (feature stride 16).
    """
    u = torch.linspace(0, grd_img_w / 16, int(grd_img_w / 16), device=device)
    v = torch.linspace(0, grd_img_h / 16, int(grd_img_h / 16), device=device)
    v, u = torch.meshgrid(v, u, indexing="ij")
    u = u.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    v = v.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return u, v


def create_spherical_grids(ground_image_size, batch_size, device):
    """Spherical angle grids over the panorama feature map.

    Args:
        ground_image_size: (H, W) panorama size, e.g. (512, 1024); the grids
            are built at the feature resolution ``H/16 x W/16``.
        batch_size: replicate the grids for this many batch elements (may
            exceed the actual loader batch; callers slice ``[:B]``).
        device: target torch device.
    Returns:
        (theta, phi), each [B, 1, H/16, W/16] radians: theta is the polar
        angle in [0, pi] (0 = zenith, pi = nadir), phi the azimuth in
        [0, 2*pi] sweeping left to right across the panorama (phi = 0 at
        the first column, forward direction in the legacy convention).
    """
    phi = torch.linspace(0, 2 * np.pi, int(ground_image_size[1] / 16),
                         device=device)
    theta = torch.linspace(0, np.pi, int(ground_image_size[0] / 16),
                           device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta = theta.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    phi = phi.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return theta, phi
