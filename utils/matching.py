"""Depth back-projection, match sampling and correspondence gathering.

These helpers connect the matcher output to the pose solvers:

1. :func:`build_matching_inputs` back-projects the ground depth map to BEV
   metric coordinates via the pinhole model and marks valid (finite-depth)
   cells.
2. :func:`sample_matches` draws correspondences from the dual-softmax
   matching probability (multinomial during training, top-k at evaluation).
3. :func:`gather_sampled_matches` looks up metric coordinates and confidence
   weights for the sampled pairs.
4. :func:`build_match_batch` chains 1-3 with a matcher forward pass.
5. :func:`get_point` converts flat match indices to feature-map UV
   coordinates plus a rescaled intrinsics matrix (visualisation /
   evaluation-time use).

Coordinate conventions:
    Camera frame:  X=right, Y=down, Z=forward (KITTI convention).
    Ground BEV:    (x, y) = (forward, right) = (Z_cam, X_cam).
    Satellite BEV: metric coordinates centred on the satellite tile.

Panorama variants (``*_panorama`` suffix, CVACT/VIGOR): the ground view is
a full panorama parameterised by spherical angles (theta, phi) instead of
pinhole intrinsics. Sampling is always multinomial (legacy behaviour, no
top-k at evaluation) and items whose entire probability row is empty are
flagged via ``valid_mask`` instead of raising.
"""
import torch
from torch.nn import functional as F


def build_matching_inputs(depth, camera_k, grd_feature_shape, metric_coord_sat,
                          u_grid, v_grid, max_depth, mask_grd=None,
                          depth_valid=True):
    """Back-project depth to ground BEV metric coordinates.

    Args:
        depth:             [B, 1, H_img, W_img] metric depth map in meters.
                           Pixels with depth >= max_depth are invalid.
        camera_k:          [B, 3, 3] intrinsics for the resized image.
        grd_feature_shape: (H/16, W/16) ground feature-map size.
        metric_coord_sat:  [B_max, N_sat, 2] satellite BEV metric coordinates.
        u_grid:            [B_max, 1, H/16, W/16] feature-cell u coordinates.
        v_grid:            [B_max, 1, H/16, W/16] feature-cell v coordinates.
        max_depth:         float, depth validity threshold in meters.
        mask_grd:          optional [B, 1, H/16, W/16] extra ground validity
                           mask (e.g. dynamic-object filtering); ANDed in.
        depth_valid:       when False the max-depth filter is skipped (the
                           coordinates still use the real depth values) so
                           evaluation can report unfiltered metrics.

    Returns:
        metric_coord_sat: [B, N_sat, 2] satellite coordinates (batch-sliced).
        bev_coord_grd:    [B, N_grd, 2] ground BEV (forward, right) meters.
        mask:             [B, N_grd] bool, True where the cell is valid.

    Raises:
        None. Invalid depths (>= max_depth) are excluded via ``mask`` rather
        than raising.
    """
    batch_size = depth.shape[0]

    # Intrinsics: image pixels -> feature-map cells (stride 16).
    camera_k = camera_k.clone()
    camera_k[:, 0, :] = camera_k[:, 0, :] / 16  # fx, cx
    camera_k[:, 1, :] = camera_k[:, 1, :] / 16  # fy, cy

    # Depth down-sampling keeps metric values (nearest), then validity mask.
    depth_downsampled = F.interpolate(depth, size=grd_feature_shape, mode="nearest")
    if depth_valid:
        mask = ~(depth_downsampled >= max_depth).flatten(1)  # [B, N_grd]
    else:
        mask = torch.ones((batch_size, grd_feature_shape[0] * grd_feature_shape[1]),
                          dtype=torch.bool, device=depth.device)
    if mask_grd is not None:
        mask_grd = mask_grd[..., 0]
        mask_grd = mask_grd.view(batch_size, -1)
        mask = torch.logical_and(mask, mask_grd.view(batch_size, -1))

    fx = camera_k[:, 0, 0].view(batch_size, 1, 1, 1)
    fy = camera_k[:, 1, 1].view(batch_size, 1, 1, 1)
    cx = camera_k[:, 0, 2].view(batch_size, 1, 1, 1)
    cy = camera_k[:, 1, 2].view(batch_size, 1, 1, 1)

    # Pinhole back-projection: X_cam = right, Z_cam = forward (Y ignored in BEV).
    grd_x = depth_downsampled                                  # forward (Z)
    grd_y = (u_grid[:batch_size] - cx) * depth_downsampled / fx  # right (X)
    grd_z = (v_grid[:batch_size] - cy) * depth_downsampled / fy  # down (Y)

    metric_coord_grd = torch.cat(
        (grd_y.flatten(2), grd_x.flatten(2), grd_z.flatten(2)), 1
    ).permute(0, 2, 1)  # [B, N_grd, (forward, right, down)]

    bev_coord_grd = metric_coord_grd[:, :, :2]  # drop the vertical axis
    return metric_coord_sat[:batch_size], bev_coord_grd, mask


def sample_matches(matching_score, num_samples_matches, training=True):
    """Sample correspondence pairs from the matching probability.

    Args:
        matching_score:      [B, N_sat, N_grd] dual-softmax probabilities.
        num_samples_matches: int, number of pairs to draw.
        training:            True -> multinomial sampling by score;
                             False -> deterministic top-k.

    Returns:
        matches_row:          [B, N_sat * N_grd] flattened probabilities.
        sampled_matching_idx: [B, K] sampled linear indices (i * N_grd + j).
        sat_indices_sampled:  [B, K] sampled satellite indices (rows).
        grd_indices_sampled:  [B, K] sampled ground indices (columns).

    Raises:
        RuntimeError: from ``torch.multinomial`` if a row of probabilities is
            all-zero (no valid matches for that sample).
    """
    _, _, num_kpts_grd = matching_score.shape
    matches_row = matching_score.flatten(1)
    if training:
        sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
    else:
        _, sampled_matching_idx = torch.topk(matches_row, num_samples_matches, dim=1)

    sat_indices_sampled = torch.div(
        sampled_matching_idx, num_kpts_grd, rounding_mode="trunc")
    grd_indices_sampled = sampled_matching_idx % num_kpts_grd
    return matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled


def gather_sampled_matches(metric_coord_sat, bev_coord_grd, matches_row,
                           sampled_matching_idx, sat_indices_sampled,
                           grd_indices_sampled):
    """Look up metric coordinates and weights for sampled match pairs.

    Args:
        metric_coord_sat:     [B, N_sat, 2] satellite metric coordinates.
        bev_coord_grd:        [B, N_grd, 2] ground BEV metric coordinates.
        matches_row:          [B, N_sat * N_grd] flattened probabilities.
        sampled_matching_idx: [B, K] sampled linear indices.
        sat_indices_sampled:  [B, K] sampled satellite indices.
        grd_indices_sampled:  [B, K] sampled ground indices.

    Returns:
        sat_points: [B, K, 2] satellite metric coordinates of the matches.
        grd_points: [B, K, 2] ground BEV coordinates of the matches.
        weights:    [B, K] matching confidence in [0, 1].
    """
    batch_size, num_samples = sampled_matching_idx.shape
    batch_idx = torch.arange(batch_size, device=metric_coord_sat.device) \
        .view(batch_size, 1).expand(-1, num_samples)
    sat_points = metric_coord_sat[batch_idx, sat_indices_sampled, :]
    grd_points = bev_coord_grd[batch_idx, grd_indices_sampled, :]
    weights = matches_row[batch_idx, sampled_matching_idx]
    return sat_points, grd_points, weights


def build_match_batch(model, depth, camera_k, grd_feature, sat_feature,
                      metric_coord_sat, u_grid, v_grid, max_depth,
                      num_samples_matches, mask_grd=None, training=True,
                      depth_valid=True):
    """One full forward pass: back-projection -> matching -> sampling.

    Args:
        model:               CrossViewMatcher instance.
        depth:               [B, 1, H_img, W_img] metric depth map.
        camera_k:            [B, 3, 3] intrinsics (resized image).
        grd_feature:         [B, C, H/16, W/16] ground feature map.
        sat_feature:         [B, C, R, R] satellite feature map.
        metric_coord_sat:    [B_max, N_sat, 2] satellite metric grid.
        u_grid, v_grid:      [B_max, 1, H/16, W/16] feature-cell coordinates.
        max_depth:           float, depth validity threshold (meters).
        num_samples_matches: int, number of matches to sample.
        mask_grd:            optional extra ground validity mask.
        training:            multinomial (train) vs top-k (eval) sampling.
        depth_valid:         False disables the max-depth validity filter
            (evaluation ``_nomask`` metrics); see :func:`build_matching_inputs`.

    Returns:
        12-tuple:
            metric_coord_sat_batch, bev_coord_grd, mask,
            sat_indices_sampled, grd_indices_sampled,
            sat_points, grd_points, weights,
            matching_score_original, matching_score,
            sat_decs, grd_decs.

        Shapes: coordinates [B, N, 2]; indices [B, K]; weights [B, K];
        matching scores [B, N_sat, N_grd]; descriptors are the matcher's
        projected features (used by InfoNCE).
    """
    metric_coord_sat_batch, bev_coord_grd, mask = build_matching_inputs(
        depth,
        camera_k,
        grd_feature.shape[-2:],
        metric_coord_sat,
        u_grid,
        v_grid,
        max_depth,
        mask_grd,
        depth_valid,
    )

    matching_score, matching_score_original, sat_decs, grd_decs = model(
        grd_feature, sat_feature, mask)

    matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled = \
        sample_matches(matching_score, num_samples_matches, training=training)

    sat_points, grd_points, weights = gather_sampled_matches(
        metric_coord_sat_batch,
        bev_coord_grd,
        matches_row,
        sampled_matching_idx,
        sat_indices_sampled,
        grd_indices_sampled,
    )

    return (
        metric_coord_sat_batch,
        bev_coord_grd,
        mask,
        sat_indices_sampled,
        grd_indices_sampled,
        sat_points,
        grd_points,
        weights,
        matching_score_original,
        matching_score,
        sat_decs,
        grd_decs,
    )


def get_point(sat_idx, grd_idx, camera_k, grd, metric_coord_sat, feat_h, feat_w):
    """Convert flat match indices to feature UV coordinates and intrinsics.

    Evaluation/visualisation helper: map sampled indices back to satellite
    metric coordinates and ground feature-map UV coordinates.

    Args:
        sat_idx:          [B, K] satellite indices into metric_coord_sat.
        grd_idx:          [B, K] flat ground indices (row * feat_w + col).
        camera_k:         [B, 3, 3] intrinsics of the original image.
        grd:              [B, C, H_img, W_img] ground image tensor (size only).
        metric_coord_sat: [B, N_sat, 2] satellite metric coordinates.
        feat_h, feat_w:   ground feature-map size (image size / 16).

    Returns:
        point_a:         [B, K, 2] ground feature (u, v) coordinates.
        point_b:         [B, K, 2] satellite metric coordinates.
        camera_k_scaled: [B, 3, 3] intrinsics scaled to feature-map pixels.
    """
    sat_expanded = sat_idx.unsqueeze(-1).expand(-1, -1, 2)
    point_b = metric_coord_sat.gather(1, sat_expanded)  # [B, K, 2]

    grd_u = grd_idx % feat_w
    grd_v = grd_idx // feat_w
    point_a = torch.stack([grd_u, grd_v], dim=-1).float()

    orig_h, orig_w = grd.shape[2], grd.shape[3]
    camera_k_scaled = camera_k.clone()
    camera_k_scaled[:, 0, :] *= (feat_w / orig_w)
    camera_k_scaled[:, 1, :] *= (feat_h / orig_h)

    return point_a, point_b, camera_k_scaled


def depth_to_metric_coordinates(depth_downsampled, theta, phi):
    """Back-project a panorama depth map to world coordinates (spherical).

    Args:
        depth_downsampled: [B, 1, H, W] metric depth (meters).
        theta: [B, 1, H, W] polar angles (0 = zenith, pi = nadir).
        phi:   [B, 1, H, W] azimuths (sweeps left to right).

    Returns:
        metric_coord_grd: [B, H*W, 3] world (X, Y, Z) with X=forward,
            Y=left, Z=up (right-handed camera frame).
        bev_coord_grd:    [B, H*W, 2] BEV coordinates (X, Y), Z dropped.
    """
    grd_x = depth_downsampled * torch.sin(theta) * torch.cos(phi)
    grd_y = depth_downsampled * torch.sin(theta) * (-torch.sin(phi))
    grd_z = depth_downsampled * torch.cos(theta)
    metric_coord_grd = torch.cat(
        (grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1
    ).permute(0, 2, 1)
    return metric_coord_grd, metric_coord_grd[:, :, :2]


def build_matching_inputs_panorama(depth, grd_feature_shape, theta, phi,
                                   max_depth, depth_valid=True):
    """Back-project panorama depth to BEV coordinates + validity mask.

    Args:
        depth:             [B, 1, H_img, W_img] panorama metric depth.
        grd_feature_shape: (H_out, W_out) ground feature-map size.
        theta, phi:        [B_max, 1, H_out, W_out] spherical angle grids
            (may carry more batch elements than ``depth``; sliced to match).
        max_depth:         float, depth validity threshold (meters).
        depth_valid:       False disables the max-depth validity filter
            (evaluation ``_nomask`` metrics); the mask becomes all-True and
            the back-projection still uses the raw depth values.

    Returns:
        bev_coord_grd: [B, H_out*W_out, 2] ground BEV metric coordinates.
        mask:          [B, H_out*W_out] bool, True where valid.
    """
    batch_size = depth.shape[0]
    depth_downsampled = F.interpolate(depth, size=grd_feature_shape,
                                      mode="nearest")
    if depth_valid:
        mask = ~(depth_downsampled >= max_depth).flatten(1)
    else:
        mask = torch.ones(batch_size, depth_downsampled.shape[2]
                          * depth_downsampled.shape[3],
                          dtype=torch.bool, device=depth.device)
    _, bev_coord_grd = depth_to_metric_coordinates(
        depth_downsampled, theta[:batch_size], phi[:batch_size])
    return bev_coord_grd, mask


def sample_matches_panorama(matching_score, num_samples_matches):
    """Multinomial match sampling with per-item validity flagging.

    Unlike the pinhole variant, sampling is always multinomial (legacy
    panorama behaviour, also at evaluation) and rows without enough
    positive mass do not raise: they are zero-padded and flagged.

    Args:
        matching_score:      [B, N_sat, N_grd] dual-softmax probabilities.
        num_samples_matches: int, number of pairs to draw.

    Returns:
        matches_row:          [B, N_sat * N_grd] flattened probabilities.
        sampled_matching_idx: [B, K] sampled linear indices (zero-padded
            for invalid items).
        sat_indices_sampled:  [B, K] sampled satellite indices.
        grd_indices_sampled:  [B, K] sampled ground indices.
        valid_mask:           [B] bool, True where the item had at least
            ``num_samples_matches`` positive entries to sample from.
    """
    _, _, num_kpts_grd = matching_score.shape
    matches_row = matching_score.flatten(1)
    matches_row = torch.nan_to_num(matches_row, nan=0.0)

    num_positive = (matches_row > 0).sum(dim=1)
    valid_mask = num_positive >= num_samples_matches

    sampled_matching_idx = torch.zeros(
        matching_score.shape[0], num_samples_matches,
        dtype=torch.long, device=matching_score.device)
    if valid_mask.any():
        sampled_matching_idx[valid_mask] = torch.multinomial(
            matches_row[valid_mask], num_samples_matches)

    sat_indices_sampled = torch.div(
        sampled_matching_idx, num_kpts_grd, rounding_mode="trunc")
    grd_indices_sampled = sampled_matching_idx % num_kpts_grd
    return (matches_row, sampled_matching_idx, sat_indices_sampled,
            grd_indices_sampled, valid_mask)


def build_match_batch_panorama(model, depth, grd_feature, sat_feature,
                               metric_coord_sat, theta, phi, max_depth,
                               num_samples_matches, depth_valid=True):
    """Panorama counterpart of :func:`build_match_batch`.

    Args:
        model:               CrossViewMatcher instance.
        depth:               [B, 1, H_img, W_img] panorama metric depth.
        grd_feature:         [B, C, H/16, W/16] ground DINO feature map.
        sat_feature:         [B, C, R, R] satellite DINO feature map.
        metric_coord_sat:    [B, N_sat, 2] satellite metric grid.
        theta, phi:          [B_max, 1, H/16, W/16] spherical angle grids.
        max_depth:           float, depth validity threshold (meters).
        num_samples_matches: int, number of matches to sample.
        depth_valid:         False disables the max-depth validity filter
            (evaluation ``_nomask`` metrics).

    Returns:
        12-tuple:
            metric_coord_sat_batch, bev_coord_grd, mask,
            sat_indices_sampled, grd_indices_sampled,
            sat_points, grd_points, weights,
            matching_score_original, sampled_matching_idx,
            valid_mask, matching_score
        with every per-item tensor filtered down to the valid subset
        (``valid_mask.all()`` is False in general, so batch size may be
        smaller than the input). ``None`` if no item is valid.
    """
    bev_coord_grd, mask = build_matching_inputs_panorama(
        depth, grd_feature.shape[-2:], theta, phi, max_depth, depth_valid)

    matching_score, matching_score_original, _, _ = model(
        grd_feature, sat_feature, mask)

    (matches_row, sampled_matching_idx, sat_indices_sampled,
     grd_indices_sampled, valid_mask) = sample_matches_panorama(
        matching_score, num_samples_matches)

    if not valid_mask.any():
        return None

    metric_coord_sat = metric_coord_sat[valid_mask]
    bev_coord_grd = bev_coord_grd[valid_mask]
    mask = mask[valid_mask]
    matches_row = matches_row[valid_mask]
    sampled_matching_idx = sampled_matching_idx[valid_mask]
    sat_indices_sampled = sat_indices_sampled[valid_mask]
    grd_indices_sampled = grd_indices_sampled[valid_mask]
    matching_score_original = matching_score_original[valid_mask]
    matching_score = matching_score[valid_mask]

    sat_points, grd_points, weights = gather_sampled_matches(
        metric_coord_sat, bev_coord_grd, matches_row,
        sampled_matching_idx, sat_indices_sampled, grd_indices_sampled)

    return (
        metric_coord_sat,
        bev_coord_grd,
        mask,
        sat_indices_sampled,
        grd_indices_sampled,
        sat_points,
        grd_points,
        weights,
        matching_score_original,
        sampled_matching_idx,
        valid_mask,
        matching_score,
    )
