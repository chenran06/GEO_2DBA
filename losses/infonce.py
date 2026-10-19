"""Bidirectional InfoNCE matching loss with GT-pose-derived correspondences.

Weak supervision: the known GT pose maps sampled satellite points into the
ground BEV frame (and vice versa); nearest neighbours become positives, and
distant cells become negatives. Both directions are averaged:

    loss = (InfoNCE_sat2grd + InfoNCE_grd2sat) / 2

The original implementation read the satellite BEV resolution from
config.ini; here ``sat_bev_res`` is an explicit argument so the loss is
config-independent. The former ``mask_sat`` satellite-validity argument has
been removed (satellite cells are always valid).
"""
import torch


def _select_first_unique_pairs(primary_indices, secondary_indices, device):
    """Keep only the first occurrence of each value in ``primary_indices``.

    Args:
        primary_indices:   [V] primary index tensor (may contain duplicates).
        secondary_indices: [V] paired secondary index tensor.
        device:            torch device for the internal zero tensor.

    Returns:
        (primary_unique, secondary_unique): [U] tensors with the first
        occurrence of each unique primary value kept, in ascending primary
        order (order follows ``torch.unique(sorted=True)``).
    """
    _, inverse, counts = torch.unique(
        primary_indices, sorted=True, return_inverse=True, return_counts=True)
    _, order = torch.sort(inverse, stable=True)
    first_positions = counts.cumsum(0)
    first_positions = torch.cat((
        torch.zeros(1, dtype=first_positions.dtype, device=device),
        first_positions[:-1],
    ))
    keep = order[first_positions]
    return primary_indices[keep], secondary_indices[keep]


def compute_infonce_loss_match_all_with_scale_select_negatives(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    mask, grid_size_h, sat_bev_res,
    negative_distance_threshold=1.0,
):
    """Bidirectional InfoNCE over the unnormalised matching score matrix.

    Args:
        Rgt:             [B, 2, 2] GT rotation (satellite -> ground frame).
        tgt:             [B, 1, 2] GT translation (meters).
        sat_points_selected: [B, K, 2] sampled satellite metric coordinates.
        grd_points_selected: [B, K, 2] sampled ground BEV coordinates.
        scale:           [B, 1] Procrustes scale (detached internally).
        sampled_row:     [B, K] sampled satellite indices.
        sampled_col:     [B, K] sampled ground indices.
        matching_score_original: [B, N_sat, N_grd] unnormalised scores
            (with -inf at invalid ground cells).
        sat_coord:       [B, N_sat, 2] all satellite metric coordinates.
        grd_coord:       [B, N_grd, 2] all ground BEV coordinates.
        mask:            [B, N_grd] bool, ground depth validity.
        grid_size_h:     scalar or [B] tensor, satellite grid physical extent
            (meters). Per-sample tensors support heterogeneous latitudes /
            tile resolutions within a batch.
        sat_bev_res:     int, satellite grid resolution per axis.
        negative_distance_threshold: float, ground points closer than this
            (scaled) are excluded from the s2g negative set (meters).

    Returns:
        [B] tensor: per-sample (s2g + g2s) / 2; callers average over the
        batch (e.g. ``torch.mean``).

    Raises:
        None. Samples without any valid positive pair contribute zero loss.
    """
    B, num_sat, num_grd = matching_score_original.shape

    scale = scale.to(torch.float32).detach()  # no gradient through the scale
    matches_row = matching_score_original.flatten(1)  # [B, N_sat * N_grd]

    if torch.is_tensor(grid_size_h):
        grid_size_h = grid_size_h.reshape(-1).to(
            matching_score_original.device, torch.float32)
    elif isinstance(grid_size_h, (list, tuple)):
        grid_size_h = torch.as_tensor(
            grid_size_h, dtype=torch.float32,
            device=matching_score_original.device).reshape(-1)
    else:
        grid_size_h = torch.full(
            (B,), float(grid_size_h),
            device=matching_score_original.device, dtype=torch.float32)
    cell_sizes = grid_size_h / sat_bev_res  # [B] physical size of one sat cell

    # ------------------------------------------------------------------
    # Direction 1: satellite -> ground.
    # ------------------------------------------------------------------
    tgt = tgt.float()
    grd_points_mapped = ((sat_points_selected - tgt.float()) / scale).float() @ Rgt.float()

    distances = torch.cdist(grd_points_mapped, grd_coord)  # [B, K, N_grd]
    min_distances, col_mapped = distances.min(dim=-1)

    grd_pointwise_distance = torch.cdist(grd_coord, grd_coord)  # [B, N_grd, N_grd]

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        # Tolerance is divided by scale because the mapping divided by scale.
        keep_index_b_distance = min_distances[b] <= cell_sizes[b] / scale[b]
        # The nearest ground cell must hold a valid (finite-depth) point.
        keep_index_b_depth = mask[b][col_mapped[b]]
        keep_index_b = keep_index_b_distance.squeeze(0) * keep_index_b_depth

        # Negative set: ground cells farther than the (scaled) threshold.
        negative_mask = grd_pointwise_distance[b] > (negative_distance_threshold / scale[b])

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique_row, unique_col_mapped = _select_first_unique_pairs(
                valid_row, valid_col_mapped, matching_score_original.device)

            negative_mask_for_unique_row = negative_mask[unique_col_mapped].to(torch.float32)
            selected_matching_indices = unique_row * num_grd + unique_col_mapped

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(
                torch.exp(matching_score_original[b, unique_row, :]) * negative_mask_for_unique_row,
                dim=1,
            ) + positives
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    # ------------------------------------------------------------------
    # Direction 2: ground -> satellite.
    # ------------------------------------------------------------------
    sat_points_mapped = scale * (grd_points_selected @ Rgt.transpose(1, 2)) + tgt

    distances = torch.cdist(sat_points_mapped, sat_coord)  # [B, K, N_sat]
    min_distances, row_mapped = distances.min(dim=-1)

    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        # No negative mask in this direction: every satellite cell counts
        # towards the denominator (asymmetric by design).
        keep_index_b = min_distances[b] <= cell_sizes[b]

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]

            unique_col, unique_row_mapped = _select_first_unique_pairs(
                valid_col, valid_row_mapped, matching_score_original.device)

            selected_matching_indices = unique_row_mapped * num_grd + unique_col

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(
                torch.exp(matching_score_original[b, :, unique_col]), dim=0)
            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2
