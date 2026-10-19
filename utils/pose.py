"""Pose solvers and validity utilities.

All functions here operate on BEV metric coordinates (meters). The Procrustes
solver maps ground-BEV points to satellite-BEV points (rotation, translation
and scale); ``e2eProbabilisticProcrustesSolver`` is the RANSAC-style variant
used at evaluation time.
"""
import torch


def estimate_pose(sat_points, grd_points, weights):
    """Weighted Procrustes pose from ground-BEV to satellite-BEV points.

    Args:
        sat_points: [B, N, 2] satellite BEV metric coordinates.
        grd_points: [B, N, 2] ground BEV metric coordinates.
        weights:    [B, N] per-correspondence confidence.
    Returns:
        (rotation [B, 2, 2], translation [B, 1, 2], scale [B, 1, 1]);
        degenerate samples carry NaN (or all-None when every sample is
        degenerate).
    """
    rotation, translation, scale, _ = weighted_procrustes_2d_with_scale(
        grd_points, sat_points, use_weights=True, use_mask=True, w=weights)
    return rotation, translation, scale


def compute_valid_pose_mask(rotation, translation, scale):
    """Boolean mask of samples whose Procrustes pose is finite.

    Args:
        rotation:     [B, 2, 2] or None.
        translation:  [B, 1, 2] or None.
        scale:        [B, 1, 1] or None.
    Returns:
        [B] bool mask; ``None`` when ``translation`` is None (all samples
        degenerate).
    """
    if translation is None:
        return None
    valid_pose = (torch.isfinite(translation).all(dim=(1, 2))
                  & torch.isfinite(rotation).all(dim=(1, 2)))
    if scale is not None:
        valid_pose &= torch.isfinite(scale).all(dim=(1, 2))
    return valid_pose


def filter_batch(valid_mask, *tensors):
    """Subset every tensor by ``valid_mask`` (None entries pass through).

    Args:
        valid_mask: [B] bool mask.
        *tensors:   tensors whose first dimension matches the batch.
    Returns:
        list of filtered tensors (same order as the inputs).
    """
    filtered = []
    for tensor in tensors:
        if tensor is None:
            filtered.append(None)
        else:
            filtered.append(tensor[valid_mask])
    return filtered


def weighted_procrustes_2d_with_scale(A, B, w=None, use_weights=True,
                                      use_mask=False, eps=1e-16, check_rank=True):
    """Weighted similarity-transform (R, t, scale) Procrustes in 2D.

    Solves  ``min sum_i w_i * ||(s * R * A_i + t) - B_i||^2``  with R in SO(2).

    Args:
        A: [B, N, 2] source points (ground BEV metric coordinates).
        B: [B, N, 2] target points (satellite BEV metric coordinates).
        w: [B, N] per-correspondence confidence weights (matching scores).
        use_weights: use ``w`` as weights; False treats all points equally
            (RANSAC minimal sets).
        use_mask: multiply raw ``w`` (instead of normalised weights) into the
            covariance; used when ``w`` is a 0/1 inlier mask.
        eps: numerical stabiliser.
        check_rank: verify cov(H) has rank >= 2; degenerate samples get NaN
            outputs instead of garbage poses.

    Returns:
        R: [B, 2, 2] rotation (ground -> satellite frame), NaN for degenerate.
        t: [B, 1, 2] translation in meters, NaN for degenerate.
        s: [B, 1, 1] scale factor, NaN for degenerate.
        ok_rank: [B] bool, True where the covariance had rank >= 2.

    Raises:
        None; if every sample is degenerate the function returns
        ``(None, None, None, ok_rank)`` early.
    """
    assert len(A) == len(B)

    # Step 1: weighted centralisation removes the translation DOF.
    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        A_mean = (w_norm * A).sum(1, keepdim=True)
        B_mean = (w_norm * B).sum(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask \
            else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean = A.mean(1, keepdim=True)
        B_mean = B.mean(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    # Step 2: rank check (collinear / single-point sets cannot fix rotation).
    if check_rank:
        ok_rank = torch.linalg.matrix_rank(H) > 1
        if not ok_rank.any():
            return None, None, None, ok_rank
    else:
        ok_rank = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)

    # Step 3: SVD -> rotation with det=+1 correction (Kabsch).
    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)

    # Step 4: closed-form scale.
    A_c_rot = A_c @ R.transpose(1, 2)
    if use_weights:
        numerator = (w_norm * B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)
        denominator = (w_norm * A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps
    else:
        numerator = (B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)
        denominator = (A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps
    s = numerator / denominator

    # Step 5: recover translation from the weighted means.
    t = B_mean - s * (A_mean @ R.transpose(1, 2))

    if check_rank and not ok_rank.all():
        R = R.clone()
        t = t.clone()
        s = s.clone()
        R[~ok_rank] = torch.nan
        t[~ok_rank] = torch.nan
        s[~ok_rank] = torch.nan

    return R, t, s, ok_rank


def soft_inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    """Differentiable (sigmoid) inlier count for a similarity-transform pose.

    Args:
        X0: [B, N, 2] source points (satellite BEV).
        X1: [B, N, 2] target points (ground BEV).
        R: [B, 2, 2] rotation.
        t: [B, 1, 2] translation.
        scale: [B, 1, 1] scale.
        th: inlier distance threshold in meters (sigmoid crosses 0.5 here).
    Returns:
        [B, 1] soft inlier count, ~1 per inlier and ~0 per far outlier.
    """
    beta = 5 / th
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)


def inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    """Hard (0/1) inlier mask for a similarity-transform pose (non-differentiable).

    Args and shapes as in :func:`soft_inlier_counting_bev_with_scale`.
    Returns:
        [B, N] float mask, 1.0 where the transformed distance <= th.
    """
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()


class e2eProbabilisticProcrustesSolver:
    """RANSAC-style probabilistic Procrustes solver (evaluation-time).

    Pipeline: multinomial sampling of correspondences by matching score ->
    minimal-set (``num_corr_2d_2d`` points) Procrustes hypotheses over
    ``it_RANSAC`` rounds -> soft-inlier scoring of every hypothesis ->
    argmax best hypothesis -> iterative hard-inlier re-weighted Procrustes
    refinement.

    Args:
        it_RANSAC: RANSAC hypothesis rounds per sampling round.
        it_matches: correspondence re-sampling rounds.
        num_samples_matches: correspondences sampled per round.
        num_corr_2d_2d: minimal-set size (3 for 4-DOF similarity).
        num_refinements: refinement iterations (inlier set must keep growing).
        th_inlier: hard inlier threshold (meters).
        th_soft_inlier: soft inlier threshold (meters).
        metric_coord_sat_B: [B, N_sat, 2] satellite metric grid.
        metric_coord_grd_B: [B, N_grd, 2] ground metric BEV coordinates.
    """

    def __init__(self, it_RANSAC, it_matches, num_samples_matches,
                 num_corr_2d_2d, num_refinements, th_inlier, th_soft_inlier,
                 metric_coord_sat_B, metric_coord_grd_B):
        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.metric_coord_sat_B = metric_coord_sat_B
        self.metric_coord_grd_B = metric_coord_grd_B

    def estimate_pose(self, matching_score, return_inliers=False):
        """Run the full sampling + RANSAC + refinement pipeline.

        Args:
            matching_score: [B, N_sat, N_grd] matching probabilities
                (detached internally; no gradients flow through).
            return_inliers: additionally return per-sample inlier tables
                (sat x, sat y, grd x, grd y, score) sorted by score.
        Returns:
            (R, t_metric, scale, best_inliers, inliers); all None when no
            valid hypothesis was found.
            R: [B, 2, 2], t_metric: [B, 1, 2] meters, scale: [B, 1, 1],
            best_inliers: [B] soft inlier count, inliers: list of [M, 5].
        """
        device = matching_score.device
        matches = matching_score.detach()
        neg_inf = torch.tensor(float('-inf'), device=device)

        B, num_kpts_sat, num_kpts_grd = matches.shape
        matches_row = matches.reshape(B, num_kpts_sat * num_kpts_grd)

        batch_idx = torch.tile(
            torch.arange(B).view(B, 1), [1, self.num_samples_matches]
        ).reshape(B, self.num_samples_matches).to(device)
        batch_idx_ransac = torch.tile(
            torch.arange(B).view(B, 1), [1, self.num_corr_2d_2d]
        ).reshape(B, self.num_corr_2d_2d).to(device)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 2, 2)).to(device)
        ts = torch.zeros((B, 0, 1, 2)).to(device)
        scales = torch.zeros((B, 0, 1, 1)).to(device)
        scores_ransac = torch.zeros((B, 0)).to(device)
        it_matches_ids = []
        dict_corr = {}

        # Stage 1+2: correspondence sampling + minimal-set hypotheses.
        for i_i in range(self.it_matches):
            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except RuntimeError:
                # Matching matrix all-zero or containing NaN.
                break

            sampled_idx_sat = torch.div(
                sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = sampled_idx % num_kpts_grd

            X = self.metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
            Y = self.metric_coord_grd_B[batch_idx, sampled_idx_grd, :]
            weights = matches_row[batch_idx, sampled_idx]
            dict_corr[i_i] = {'X': X, 'Y': Y, 'weights': weights}

            for _ in range(self.it_RANSAC):
                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_2d_2d)
                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]

                # Equal-weight Procrustes on the minimal set.
                R, t, scale, ok_rank = weighted_procrustes_2d_with_scale(
                    Y_k, X_k, use_weights=False)
                if t is None:
                    continue

                valid_h = ok_rank.clone()
                valid_h &= torch.isfinite(t).all(dim=(1, 2))
                valid_h &= torch.isfinite(R).all(dim=(1, 2))
                valid_h &= torch.isfinite(scale).all(dim=(1, 2))
                if not valid_h.any():
                    continue

                # Stage 3: soft inlier scoring on the full sampled set.
                score_k = soft_inlier_counting_bev_with_scale(
                    Y, X, R, t, scale, th=self.th_soft_inlier)
                score_k = score_k.masked_fill(~valid_h.unsqueeze(-1), neg_inf)

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scales = torch.cat([scales, scale.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h == 0:
            return None, None, None, None, None

        has_valid = torch.isfinite(scores_ransac).any(dim=1)
        if not has_valid.any():
            return None, None, None, None, None

        # Stage 4: keep the hypothesis with the highest soft inlier count.
        row_idx = torch.arange(B, device=device)
        max_ind = torch.argmax(scores_ransac, dim=1)
        R = Rs[row_idx, max_ind].clone()
        t_metric = ts[row_idx, max_ind].clone()
        scale = scales[row_idx, max_ind].clone()
        best_inliers = scores_ransac[row_idx, max_ind].clone()

        X_best = torch.zeros_like(X)
        Y_best = torch.zeros_like(Y)
        for i_b in range(B):
            if not has_valid[i_b]:
                continue
            corr = dict_corr[it_matches_ids[int(max_ind[i_b])]]
            X_best[i_b], Y_best[i_b] = corr['X'][i_b], corr['Y'][i_b]

        # Stage 5: iterative hard-inlier re-weighted Procrustes refinement.
        inliers_ref = torch.zeros((B, self.num_samples_matches)).to(device)
        th_ref = self.num_refinements * [self.th_inlier]
        inliers_pre = self.num_corr_2d_2d * torch.ones_like(best_inliers)

        for i_ref in range(len(th_ref)):
            if not has_valid.any():
                break

            inliers = torch.zeros((B, self.num_samples_matches), device=device)
            inliers[has_valid] = inlier_counting_bev_with_scale(
                Y_best[has_valid], X_best[has_valid],
                R[has_valid], t_metric[has_valid], scale[has_valid],
                th=th_ref[i_ref],
            )

            do_ref = (has_valid
                      & (inliers.sum(-1) >= self.num_corr_2d_2d)
                      & (inliers.sum(-1) > inliers_pre))
            inliers_pre[do_ref] = inliers.sum(-1)[do_ref]
            if do_ref.sum().item() == 0:
                break

            inliers_ref[do_ref] = inliers[do_ref]
            R_ref, t_ref, scale_ref, _ = weighted_procrustes_2d_with_scale(
                Y_best[do_ref], X_best[do_ref],
                use_weights=True, use_mask=True, check_rank=False,
                w=inliers_ref[do_ref],
            )
            R[do_ref], t_metric[do_ref], scale[do_ref] = R_ref, t_ref, scale_ref

        best_inliers = torch.full_like(best_inliers, torch.nan)
        best_inliers[has_valid] = soft_inlier_counting_bev_with_scale(
            Y_best[has_valid], X_best[has_valid],
            R[has_valid], t_metric[has_valid], scale[has_valid],
            th=self.th_inlier,
        ).squeeze(-1)

        R[~has_valid] = torch.nan
        t_metric[~has_valid] = torch.nan
        scale[~has_valid] = torch.nan

        inliers = None
        if return_inliers:
            inliers = self._collect_inliers(
                B, has_valid, max_ind, dict_corr, it_matches_ids,
                X_best, Y_best, R, t_metric, scale,
            )

        return R, t_metric, scale, best_inliers, inliers

    def _collect_inliers(self, B, has_valid, max_ind, dict_corr,
                         it_matches_ids, X_best, Y_best, R, t_metric, scale):
        """Assemble per-sample inlier tables sorted by matching score."""
        weights_best = torch.zeros_like(X_best[..., 0])
        for i_b in range(B):
            if not has_valid[i_b]:
                continue
            corr = dict_corr[it_matches_ids[int(max_ind[i_b])]]
            X_best[i_b], Y_best[i_b] = corr['X'][i_b], corr['Y'][i_b]
            weights_best[i_b] = corr['weights'][i_b]

        inliers_idxs = torch.zeros(
            (B, self.num_samples_matches), device=X_best.device)
        inliers_idxs[has_valid] = inlier_counting_bev_with_scale(
            Y_best[has_valid], X_best[has_valid],
            R[has_valid], t_metric[has_valid], scale[has_valid],
            th=self.th_inlier,
        )

        inliers = []
        for idx_b in range(B):
            if not has_valid[idx_b]:
                inliers.append(torch.empty((0, 5), device=X_best.device))
                continue
            X_inliers = X_best[idx_b, inliers_idxs[idx_b] == 1.]
            Y_inliers = Y_best[idx_b, inliers_idxs[idx_b] == 1.]
            score_inliers = weights_best[idx_b, inliers_idxs[idx_b] == 1.]
            order_corr = torch.argsort(score_inliers, descending=True)
            inliers.append(torch.cat([
                X_inliers[order_corr],
                Y_inliers[order_corr],
                score_inliers[order_corr].unsqueeze(-1),
            ], dim=1))
        return inliers
