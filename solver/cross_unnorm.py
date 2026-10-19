"""Unified cross_unnorm pose solver (clockwise convention, single module).

All three solvers share one internal convention ("clockwise", cw):

    residual (perpendicular distance):
        cross_i(psi) = dx_i * sin(psi + theta_i) + dy_i * cos(psi + theta_i)
    signed depth (front > 0):
        t_i(psi)     = dx_i * cos(psi + theta_i) - dy_i * sin(psi + theta_i)
    with dx, dy = sat_i - C and C the camera position in metric BEV.

Given psi, the loss is quadratic in C, giving the closed form C(psi)
(:func:`_eval_grid`); psi is found by search. The objective is 180-degree
periodic in psi; the e2e solver breaks the tie with a cheirality penalty
on the signed depth (front points must have t > 0), so its psi output is
on the physically correct branch.

Bridging from the legacy pinhole convention: the legacy solvers
(``uilt/cross_unnorm.py`` v1 bisection, ``uilt/cross_unnorm_v4_e2e.py``)
use theta_v1 = atan((u - cx) / fx) and residual dx*cos(psi+theta) -
dy*sin(psi+theta). With theta_cw = theta_v1 - pi/2 the cw residual equals
minus the v1 residual (same square), the signed depth is unchanged and the
closed-form C(psi) is bit-for-bit the same linear system. Hence
:func:`compute_theta_pinhole` returns the bridged angle and every solver
below reproduces the legacy pinhole results at the same psi values.
Panoramic inputs (CVACT/VIGOR) pass their azimuth theta directly, matching
the legacy ``uilt/cross_unnorm_v2_clockwise*.py`` family.

Three entry points:

1. :func:`solve_e2e`       differentiable coarse-to-fine soft-argmin
                           (training-time; participates in gradients).
2. :func:`solve_bisection` non-differentiable coarse argmin + bisection,
                           per-sample under no_grad (evaluation / metrics).
3. :func:`solve_ransac`    RANSAC-style probabilistic solver with soft
                           inlier scoring and hard-inlier refinement
                           (evaluation / metrics).

Default parameters match the legacy scripts; callers should pass
coarse_step_deg / cheir_lambda explicitly per dataset:
KITTI & RobotCar use coarse_step_deg=10.0, cheir_lambda=10.0;
CVACT & VIGOR use coarse_step_deg=2.5, cheir_lambda=1.0.
"""
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════
# Theta computation (pinhole -> cw bridging)
# ═══════════════════════════════════════════════════════════════

def compute_theta_pinhole(grd_uv, camera_k_scaled, feat_w):
    """Ray angles for pinhole ground images, in the cw convention.

    Reproduces the legacy u-coordinate lookup (u' = linspace(0, feat_w,
    feat_w)[u_int], i.e. u' = u_int * feat_w / (feat_w - 1)) before
    atan((u' - cx) / fx), then subtracts pi/2 to bridge into the cw
    convention (see module docstring).

    Args:
        grd_uv:          [B, N, 2] ground feature-cell UV coordinates (int).
        camera_k_scaled: [B, 3, 3] intrinsics in feature-map pixels
                         (original intrinsics / 16).
        feat_w:          ground feature-map width in cells (e.g. 88 for
                         KITTI 1408 px images with stride 16).

    Returns:
        theta: [B, N] ray angles in radians (cw convention).
    """
    device = grd_uv.device
    fx = camera_k_scaled[:, 0, 0]
    cx = camera_k_scaled[:, 0, 2]
    u_int = grd_uv[:, :, 0].float()
    u_grid = torch.linspace(0, feat_w, feat_w, device=device)
    u_linspace = u_grid[u_int.clamp(0, feat_w - 1).long()]
    theta_v1 = torch.atan(
        (u_linspace - cx.unsqueeze(1)) / fx.unsqueeze(1).clamp(min=1e-8))
    return theta_v1 - np.pi / 2


# ═══════════════════════════════════════════════════════════════
# Internal helpers (cw convention)
# ═══════════════════════════════════════════════════════════════

def _normalize_weights(weights):
    """Row-normalise [B, N] weights to sum to 1; None passes through."""
    if weights is None:
        return None
    return weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)


def _eval_grid(sat_points, psi_grid, theta, w):
    """Closed-form C(psi), cross loss and cheirality penalty on a psi grid.

    Args:
        sat_points: [B, N, 2] satellite BEV points (meters).
        psi_grid:   [K] or [B, K] candidate yaw values (radians).
        theta:      [B, N] ray angles (radians, cw convention).
        w:          [B, N] weights summing to 1 per row.

    Returns:
        C_grid: [B, K, 2] closed-form camera position for each psi.
        losses: [B, K]    cross loss L(psi) with C = C(psi).
        cheir:  [B, K]    cheirality penalty P(psi) in [0, 1].
        det:    [B, K]    determinant of the 2x2 normal equations
                          (degeneracy check for minimal sets).
    """
    B = sat_points.shape[0]
    if psi_grid.dim() == 1:
        psi_grid = psi_grid.view(1, -1).expand(B, -1)

    psi_b = psi_grid.unsqueeze(2)                    # [B, K, 1]
    theta_b = theta.unsqueeze(1)                     # [B, 1, N]
    w_b = w.unsqueeze(1)                             # [B, 1, N]
    c_i = torch.cos(psi_b + theta_b)                 # [B, K, N]
    s_i = torch.sin(psi_b + theta_b)

    wc2 = (w_b * c_i * c_i).sum(dim=2)               # [B, K]
    ws2 = (w_b * s_i * s_i).sum(dim=2)
    wcs = (w_b * c_i * s_i).sum(dim=2)
    sx = sat_points[:, :, 0].unsqueeze(1)            # [B, 1, N]
    sy = sat_points[:, :, 1].unsqueeze(1)
    b0 = (w_b * sx * s_i * s_i).sum(dim=2) \
       + (w_b * sy * s_i * c_i).sum(dim=2)           # [B, K]
    b1 = (w_b * sx * s_i * c_i).sum(dim=2) \
       + (w_b * sy * c_i * c_i).sum(dim=2)           # [B, K]
    det = wc2 * ws2 - wcs * wcs + 1e-8
    C_x = (wc2 * b0 - wcs * b1) / det
    C_y = (-wcs * b0 + ws2 * b1) / det
    C_grid = torch.stack([C_x, C_y], dim=-1)         # [B, K, 2]

    dx = sx - C_x.unsqueeze(-1)                      # [B, K, N]
    dy = sy - C_y.unsqueeze(-1)
    cross = dx * s_i + dy * c_i                      # cw perpendicular distance
    losses = (w_b * cross * cross).sum(dim=2)        # [B, K]

    # Cheirality: physical ray is (c, -s); signed depth t = dx*c - dy*s > 0
    # for front points. psi -> psi + 180deg flips every t, so the penalty
    # breaks the two-branch symmetry.
    t = dx * c_i - dy * s_i
    pen_num = (w_b * torch.relu(-t) ** 2).sum(dim=2)
    pen_den = (w_b * t * t).sum(dim=2) + 1e-8
    cheir = pen_num / pen_den                        # [B, K]
    return C_grid, losses, cheir, det


def _eval_pose(sat_points, C, psi, theta, w):
    """Cross loss and cheirality penalty at an explicit pose (C, psi).

    Args:
        sat_points: [B, N, 2]; C: [B, 2]; psi: [B]; theta: [B, N]; w: [B, N].
    Returns:
        loss:  [B] cross loss.
        cheir: [B] cheirality penalty in [0, 1].
    """
    s_i = torch.sin(psi.unsqueeze(1) + theta)        # [B, N]
    c_i = torch.cos(psi.unsqueeze(1) + theta)
    dx = sat_points[..., 0] - C[:, 0:1]
    dy = sat_points[..., 1] - C[:, 1:2]
    cross = dx * s_i + dy * c_i
    loss = (w * cross * cross).sum(dim=1)            # [B]
    t = dx * c_i - dy * s_i
    cheir = (w * torch.relu(-t) ** 2).sum(dim=1) \
          / ((w * t * t).sum(dim=1) + 1e-8)
    return loss, cheir


def cross_unnorm_loss(C, psi, sat_points, theta, weights=None):
    """Weighted cross_unnorm residual at a candidate pose (cw convention).

    The solver objective, also usable as the GT-pose monitoring loss.

    Args:
        C:          [B, 2] camera position in metric BEV (meters).
        psi:        [B] yaw angle (radians, cw convention).
        sat_points: [B, N, 2] satellite BEV points (meters).
        theta:      [B, N] ray angles (radians, cw convention).
        weights:    [B, N] matching weights (None -> uniform).

    Returns:
        loss: [B] per-sample weighted cross loss.
    """
    w = _normalize_weights(weights)
    if w is None:
        w = torch.ones_like(theta) / theta.shape[1]

    psi_b = psi.view(-1, 1)
    c_i = torch.cos(psi_b + theta)
    s_i = torch.sin(psi_b + theta)

    delta = sat_points - C.view(-1, 1, 2)
    cross_val = delta[:, :, 0] * s_i + delta[:, :, 1] * c_i
    return (w * cross_val * cross_val).sum(dim=1)


# ═══════════════════════════════════════════════════════════════
# Solver 1: differentiable e2e soft-argmin
# ═══════════════════════════════════════════════════════════════

def _solve_one(sat, theta, w, coarse_step_deg, fine_tol_deg,
               coarse_tau, fine_tau, cheir_lambda, dual_window):
    """Coarse-to-fine soft-argmin for one (small) batch, fully differentiable.

    Stage 1: full-circle coarse scan, soft argmin under the cheirality-
    weighted objective (branch selection).
    Stage 2: fine scan(s) with pure L around the coarse weighted centre
    (dual_window additionally opens a second window at the coarse argmin
    grid point to guard against near-tie samples).
    Stage 3 (dual_window only): soft-merge the two candidates under the
    cheirality-weighted objective.
    """
    device, dtype = sat.device, sat.dtype
    B = sat.shape[0]

    coarse_psi = torch.deg2rad(torch.arange(
        -180.0, 180.0, coarse_step_deg, device=device, dtype=dtype))  # [Kc]
    _, losses_c, cheir_c, _ = _eval_grid(sat, coarse_psi, theta, w)
    Lt_c = losses_c + cheir_lambda * cheir_c                          # [B, Kc]
    w_c = torch.softmax(-Lt_c / coarse_tau, dim=1)
    psi_blend = (w_c * coarse_psi.view(1, -1)).sum(dim=1)             # [B]

    if dual_window:
        # Coarse argmin grid point (constant, no gradient needed).
        i_top = Lt_c.argmin(dim=1)                                    # [B]
        psi_top = coarse_psi.view(1, -1).expand(B, -1).gather(
            1, i_top.view(-1, 1)).squeeze(1)                          # [B]
        centers = (psi_blend, psi_top)
        half = max(np.deg2rad(coarse_step_deg), np.deg2rad(5.0))
    else:
        centers = (psi_blend,)
        half = np.deg2rad(coarse_step_deg)

    step_f = np.deg2rad(fine_tol_deg)
    Kf = min(512, max(8, int(np.ceil(2 * half / step_f))))
    offsets = torch.linspace(-half, half, Kf, device=device, dtype=dtype)

    cands = []
    for center in centers:
        fine_psi = center.unsqueeze(1) + offsets.unsqueeze(0)         # [B, Kf]
        C_grid_f, losses_f, _, _ = _eval_grid(sat, fine_psi, theta, w)
        w_f = torch.softmax(-losses_f / fine_tau, dim=1)
        cands.append(((w_f.unsqueeze(-1) * C_grid_f).sum(dim=1),      # C [B, 2]
                      (w_f * fine_psi).sum(dim=1)))                   # psi [B]

    if dual_window:
        Lt = torch.stack(
            [L_i + cheir_lambda * P_i
             for L_i, P_i in (_eval_pose(sat, C_i, psi_i, theta, w)
                              for C_i, psi_i in cands)], dim=1)       # [B, 2]
        beta = torch.softmax(-Lt / fine_tau, dim=1)                   # [B, 2]
        C_pred = beta[:, 0:1] * cands[0][0] + beta[:, 1:2] * cands[1][0]
        psi_pred = beta[:, 0] * cands[0][1] + beta[:, 1] * cands[1][1]
    else:
        C_pred, psi_pred = cands[0]
    return C_pred, psi_pred


def solve_e2e(sat_points, theta, weights=None, coarse_step_deg=10.0,
              fine_tol_deg=0.01, coarse_tau=0.05, fine_tau=0.005,
              cheir_lambda=10.0, dual_window=False, chunk_size=None):
    """Differentiable coarse-to-fine solve without 180-degree ambiguity.

    Args:
        sat_points:      [B, N, 2] satellite BEV points (meters).
        theta:           [B, N] ray angles (radians, cw convention).
        weights:         [B, N] matching weights (None -> uniform).
        coarse_step_deg: coarse grid step in degrees, full circle
                         [-180, 180).
        fine_tol_deg:    fine grid resolution in degrees.
        coarse_tau:      coarse softmax temperature (rad).
        fine_tau:        fine softmax temperature (rad).
        cheir_lambda:    cheirality penalty weight (coarse branch choice).
        dual_window:     True -> second fine window at the coarse argmin +
                         candidate merge (legacy panorama behaviour);
                         False -> single window (legacy KITTI behaviour).
        chunk_size:      optional int, process the batch in chunks of this
                         size to bound memory (per-sample independent).

    Returns:
        C_pred:   [B, 2] camera position (meters).
        psi_pred: [B] yaw (radians, cw convention, correct branch).
        loss:     [B] cross loss at (C_pred, psi_pred).
    """
    B = sat_points.shape[0]
    w = _normalize_weights(weights)
    if w is None:
        w = torch.ones_like(theta) / theta.shape[1]

    if chunk_size is None or B <= chunk_size:
        C_pred, psi_pred = _solve_one(
            sat_points, theta, w, coarse_step_deg, fine_tol_deg,
            coarse_tau, fine_tau, cheir_lambda, dual_window)
    else:
        Cs, Ps = [], []
        for i in range(0, B, chunk_size):
            C_i, psi_i = _solve_one(
                sat_points[i:i + chunk_size], theta[i:i + chunk_size],
                w[i:i + chunk_size], coarse_step_deg, fine_tol_deg,
                coarse_tau, fine_tau, cheir_lambda, dual_window)
            Cs.append(C_i)
            Ps.append(psi_i)
        C_pred = torch.cat(Cs, dim=0)
        psi_pred = torch.cat(Ps, dim=0)

    loss = cross_unnorm_loss(C_pred, psi_pred, sat_points, theta, weights)
    return C_pred, psi_pred, loss


# ═══════════════════════════════════════════════════════════════
# Solver 2: non-differentiable coarse argmin + bisection
# ═══════════════════════════════════════════════════════════════

def solve_bisection(sat_points, theta, weights=None, coarse_step_deg=10.0,
                    fine_tol_deg=0.01, verbose=False):
    """Hard coarse-argmin + bisection solve (no gradients, per-sample).

    For each candidate psi: closed-form C(psi), evaluate the loss; coarse
    scan at ``coarse_step_deg``, then binary-search within +-coarse_step
    down to ``fine_tol_deg``. Wrapped in no_grad; matches the legacy v1 /
    v2 bisection solvers exactly.

    Args:
        sat_points:      [B, N, 2] satellite BEV points (meters).
        theta:           [B, N] ray angles (radians, cw convention).
        weights:         [B, N] matching weights (None -> uniform).
        coarse_step_deg: coarse grid step in degrees, full circle
                         [-180, 180).
        fine_tol_deg:    bisection tolerance in degrees.
        verbose:         print progress every 20 samples.

    Returns:
        C_pred:   [B, 2] camera position (meters).
        psi_pred: [B] yaw (radians, cw convention).
        loss:     [B] cross loss at the returned pose.
    """
    B = sat_points.shape[0]
    device = sat_points.device
    dtype = sat_points.dtype

    with torch.no_grad():
        w = _normalize_weights(weights)
        if w is None:
            w = torch.ones_like(theta) / theta.shape[1]

        coarse_deg = torch.arange(-180, 180, coarse_step_deg,
                                  device=device, dtype=dtype)
        coarse_psi = torch.deg2rad(coarse_deg)
        tol_rad = np.deg2rad(fine_tol_deg)
        half_window_rad = np.deg2rad(coarse_step_deg)

        psi_pred = torch.zeros(B, device=device, dtype=dtype)
        C_pred = torch.zeros(B, 2, device=device, dtype=dtype)
        loss_pred = torch.zeros(B, device=device, dtype=dtype)

        def _eval(psi_scalar, sat_i, theta_i, w_i):
            """Loss of the closed-form pose at one psi value (scalar float)."""
            psi_t = torch.tensor([psi_scalar], device=device,
                                 dtype=dtype).view(1)
            _, losses, _, _ = _eval_grid(sat_i, psi_t, theta_i, w_i)
            return losses[0, 0].item()

        for b in range(B):
            sat_i = sat_points[b:b + 1]
            theta_i = theta[b:b + 1]
            w_i = w[b:b + 1]

            # Stage 1: coarse scan, track only the best psi (scalar).
            best_loss = float('inf')
            best_psi = 0.0
            for psi_val in coarse_psi:
                loss_val = _eval(psi_val.item(), sat_i, theta_i, w_i)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_psi = psi_val.item()

            # Stage 2: bisection within +-coarse_step around the argmin.
            left = best_psi - half_window_rad
            right = best_psi + half_window_rad
            while right - left > tol_rad:
                mid = (left + right) / 2
                eps = tol_rad
                if _eval(mid - eps, sat_i, theta_i, w_i) \
                        < _eval(mid + eps, sat_i, theta_i, w_i):
                    right = mid
                else:
                    left = mid
            best_psi = (left + right) / 2

            # Final C and loss at the found psi.
            psi_t = torch.tensor([best_psi], device=device,
                                 dtype=dtype).view(1)
            C_grid, losses, _, _ = _eval_grid(sat_i, psi_t, theta_i, w_i)

            psi_pred[b] = best_psi
            C_pred[b] = C_grid[0, 0]
            loss_pred[b] = losses[0, 0]

            if verbose and b % 20 == 0:
                print(f"  solve_bisection sample {b:3d}/{B}: "
                      f"psi={np.rad2deg(best_psi):.2f} deg, "
                      f"loss={losses[0, 0].item():.6f}")

    return C_pred, psi_pred, loss_pred


# ═══════════════════════════════════════════════════════════════
# Solver 3: RANSAC-style probabilistic solver
# ═══════════════════════════════════════════════════════════════

class e2eProbabilisticCrossUnnormSolver:
    """RANSAC cross_unnorm solver (batched, no gradients, e2e-Procrustes style).

    Pipeline: probability sampling of correspondences -> minimal-set
    (``num_corr_2d_2d`` points) hypotheses via psi grid scan + closed-form
    C -> soft inlier scoring -> argmax best hypothesis -> iterative
    hard-inlier re-weighted psi-scan refinement. Degenerate samples fall
    back to :func:`solve_bisection` on the full point set.

    Args:
        it_RANSAC:           hypothesis rounds per sampling round.
        it_matches:          correspondence re-sampling rounds.
        num_samples_matches: correspondences sampled per round
                             (None -> all N points).
        num_corr_2d_2d:      minimal-set size (3 for the 3-DOF pose).
        num_refinements:     refinement iterations (inlier count must grow).
        th_inlier:           hard inlier threshold (meters).
        th_soft_inlier:      soft inlier threshold (meters).
        minimal_step_deg:    psi scan step for minimal-set hypotheses.
        coarse_step_deg:     coarse scan step for refinement + fallback.
        fine_step_deg:       fine scan step for refinement.
        det_eps:             degeneracy threshold on the normal-equation
                             determinant.
        refine_weight_mode:  'mask' = 0/1 inlier weights;
                             'mask*weights' = inlier mask x matching score.
        seed:                optional RNG seed for reproducibility.
    """

    def __init__(self, it_RANSAC=100, it_matches=1, num_samples_matches=None,
                 num_corr_2d_2d=3, num_refinements=2, th_inlier=5.0,
                 th_soft_inlier=5.0, minimal_step_deg=1.0,
                 coarse_step_deg=10.0, fine_step_deg=0.1, det_eps=1e-6,
                 refine_weight_mode='mask*weights', seed=None):
        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.minimal_step_deg = minimal_step_deg
        self.coarse_step_deg = coarse_step_deg
        self.fine_step_deg = fine_step_deg
        self.det_eps = det_eps
        assert refine_weight_mode in ('mask', 'mask*weights')
        self.refine_weight_mode = refine_weight_mode
        self.seed = seed

    @staticmethod
    def _residual(sat, theta, C, psi):
        """Signed cw perpendicular distance; [B, S, 2]/[B, S]/[B, 2]/[B] -> [B, S]."""
        c = torch.cos(psi.unsqueeze(-1) + theta)
        s = torch.sin(psi.unsqueeze(-1) + theta)
        dx = sat[..., 0] - C[:, 0:1]
        dy = sat[..., 1] - C[:, 1:2]
        return dx * s + dy * c

    def _soft_inlier(self, sat, theta, C, psi):
        """Soft inlier count: sum of sigmoid(5/th * (th - |r_i|))."""
        r = self._residual(sat, theta, C, psi).abs()      # [B, S]
        beta = 5.0 / self.th_soft_inlier
        return torch.sigmoid(beta * (self.th_soft_inlier - r)).sum(dim=-1)

    def estimate_pose(self, sat_points, theta, weights=None):
        """Run the full sampling + RANSAC + refinement pipeline.

        Args:
            sat_points: [B, N, 2] satellite BEV points (meters).
            theta:      [B, N] ray angles (radians, cw convention).
            weights:    [B, N] matching weights (None -> uniform).

        Returns:
            C_pred:      [B, 2] camera position (meters).
            psi_pred:    [B] yaw (radians, cw convention).
            loss:        [B] cross loss on the full point set at the final
                         pose (normalised weights, dead-fixed).
            inlier_mask: [B, N] bool, hard inliers at the final pose
                         (True everywhere for fallback samples).

        Raises:
            ValueError: if ``num_samples_matches`` < ``num_corr_2d_2d``.
        """
        device = sat_points.device
        B, N, _ = sat_points.shape
        dtype = sat_points.dtype

        # Dedicated generator when seeded: reproducible without perturbing the
        # global RNG stream (the training loop draws match samples from it).
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
        if self.num_samples_matches is not None \
                and self.num_samples_matches < self.num_corr_2d_2d:
            raise ValueError("num_samples_matches < num_corr_2d_2d")

        sat = sat_points.detach()
        th_all = theta.detach()

        if weights is None:
            w_all = torch.ones(B, N, device=device, dtype=dtype) / N
        else:
            w_all = weights.detach().float()
            w_all = w_all / w_all.sum(dim=1, keepdim=True).clamp(min=1e-8)
        dead = w_all.sum(dim=1) < 1e-12
        if dead.any():
            w_all[dead] = 1.0 / N  # multinomial needs non-zero weights

        bidx = torch.arange(B, device=device).unsqueeze(1)           # [B, 1]
        neg_inf = float('-inf')

        # Stage 1: probability sampling of the correspondence set.
        if self.num_samples_matches is not None \
                and self.num_samples_matches < N:
            S, use_multinomial = self.num_samples_matches, True
        else:
            S, use_multinomial = N, False

        dict_corr = {}
        hyp_C, hyp_psi, hyp_score = [], [], []
        it_matches_ids = []

        psi_min_grid = torch.deg2rad(torch.arange(
            -180.0, 180.0, self.minimal_step_deg, device=device, dtype=dtype))

        for i_m in range(self.it_matches):
            if use_multinomial:
                idx = torch.multinomial(w_all, S, replacement=False,
                                        generator=gen)  # [B, S]
            else:
                idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

            sat_r = sat[bidx, idx]          # [B, S, 2]
            th_r = th_all[bidx, idx]        # [B, S]
            w_r = w_all[bidx, idx]          # [B, S]
            dict_corr[i_m] = {'sat': sat_r, 'theta': th_r, 'w': w_r, 'idx': idx}

            # Stage 2+3: minimal-set hypotheses + soft inlier scoring
            # (each round generates one hypothesis per batch element).
            for _ in range(self.it_RANSAC):
                sel = torch.multinomial(w_r, self.num_corr_2d_2d,
                                        generator=gen)                # [B, m]
                sat_m = sat_r[bidx, sel]                              # [B, m, 2]
                th_m = th_r[bidx, sel]                                # [B, m]
                w_m = w_r[bidx, sel]                                  # [B, m]
                w_m = w_m / w_m.sum(dim=1, keepdim=True).clamp(min=1e-8)

                C_g, L_g, _, det_g = _eval_grid(sat_m, psi_min_grid, th_m, w_m)
                k = L_g.argmin(dim=-1)                                # [B]
                C_h = torch.gather(C_g, 1,
                                   k.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
                psi_h = psi_min_grid[k]                               # [B]
                det_h = torch.gather(det_g, 1, k.unsqueeze(1)).squeeze(1)

                # det is the weighted covariance of (c, s) directions; it
                # only depends on the theta distribution, so det ~ 0 means
                # a degenerate (collinear-in-angle) minimal set.
                valid = (det_h > self.det_eps) & torch.isfinite(C_h).all(dim=-1)

                score = self._soft_inlier(sat_r, th_r, C_h, psi_h)    # [B]
                score = score.masked_fill(~valid, neg_inf)

                hyp_C.append(C_h)
                hyp_psi.append(psi_h)
                hyp_score.append(score)
                it_matches_ids.append(i_m)

        hyp_C = torch.stack(hyp_C, dim=1)          # [B, T, 2]
        hyp_psi = torch.stack(hyp_psi, dim=1)      # [B, T]
        hyp_score = torch.stack(hyp_score, dim=1)  # [B, T]

        # Stage 4: keep the hypothesis with the highest soft inlier count.
        has_valid = torch.isfinite(hyp_score).any(dim=1)              # [B]
        max_ind = hyp_score.argmax(dim=1)                             # [B]
        row = torch.arange(B, device=device)
        C = hyp_C[row, max_ind].clone()                               # [B, 2]
        psi = hyp_psi[row, max_ind].clone()                           # [B]

        # Point set of the best hypothesis's sampling round.
        X_best = torch.zeros(B, S, 2, device=device, dtype=dtype)
        TH_best = torch.zeros(B, S, device=device, dtype=dtype)
        W_best = torch.zeros(B, S, device=device, dtype=dtype)
        IDX_best = torch.zeros(B, S, device=device, dtype=torch.long)
        for b in range(B):
            if not has_valid[b]:
                continue
            r_ = dict_corr[it_matches_ids[int(max_ind[b])]]
            X_best[b] = r_['sat'][b]
            TH_best[b] = r_['theta'][b]
            W_best[b] = r_['w'][b]
            IDX_best[b] = r_['idx'][b]

        # Stage 5: iterative hard-inlier re-weighted refinement
        # (inlier count must stay >= minimal set and keep growing).
        inliers_pre = self.num_corr_2d_2d * torch.ones(
            B, device=device, dtype=dtype)
        for _ in range(self.num_refinements):
            if not has_valid.any():
                break
            r = self._residual(X_best[has_valid], TH_best[has_valid],
                               C[has_valid], psi[has_valid]).abs()    # [Bv, S]
            inl = (r < self.th_inlier).float()
            inl_full = torch.zeros(B, S, device=device, dtype=dtype)
            inl_full[has_valid] = inl
            cnt = inl_full.sum(dim=-1)

            do_ref = has_valid & (cnt >= self.num_corr_2d_2d) \
                & (cnt > inliers_pre)
            inliers_pre[do_ref] = cnt[do_ref]
            if not do_ref.any():
                break

            w_mask = inl_full[do_ref] \
                / inl_full[do_ref].sum(dim=1, keepdim=True).clamp(min=1e-8)
            if self.refine_weight_mode == 'mask*weights':
                w_raw = inl_full[do_ref] * W_best[do_ref]
                s_raw = w_raw.sum(dim=1, keepdim=True)
                w_in = torch.where(s_raw > 1e-12,
                                   w_raw / s_raw.clamp(min=1e-12), w_mask)
            else:
                w_in = w_mask
            C_ref, psi_ref = self._solve_scan(
                X_best[do_ref], TH_best[do_ref], w_in)
            C[do_ref] = C_ref
            psi[do_ref] = psi_ref

        # Hard inlier mask at the final pose, mapped back to input order.
        inlier_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if has_valid.any():
            r_final = self._residual(X_best[has_valid], TH_best[has_valid],
                                     C[has_valid], psi[has_valid]).abs()
            mask_v = r_final < self.th_inlier
            hv_idx = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
            inlier_mask[hv_idx.unsqueeze(1), IDX_best[hv_idx]] = mask_v

        # Degenerate samples: fall back to the bisection solver on the
        # full point set (legacy returned NaN here; this is more useful).
        if not has_valid.all():
            fb = ~has_valid
            C_fb, psi_fb, _ = solve_bisection(
                sat_points[fb], th_all[fb],
                weights[fb] if weights is not None else None,
                coarse_step_deg=self.coarse_step_deg)
            C[fb] = C_fb
            psi[fb] = psi_fb
            inlier_mask[fb] = True

        # Final cross loss (drop-in third return value, matches solve_e2e).
        c_i = torch.cos(psi.unsqueeze(1) + th_all)
        s_i = torch.sin(psi.unsqueeze(1) + th_all)
        dx = sat[..., 0] - C[:, 0:1]
        dy = sat[..., 1] - C[:, 1:2]
        cross_val = dx * s_i + dy * c_i
        loss = (w_all * cross_val * cross_val).sum(dim=1)

        return C, psi, loss, inlier_mask

    def _solve_scan(self, sat, theta, w):
        """Coarse + fine hard argmin scan for refinement (batched).

        Args:
            sat: [Bv, S, 2]; theta: [Bv, S]; w: [Bv, S] (sums to 1).
        Returns:
            C: [Bv, 2]; psi: [Bv].
        """
        Bv = sat.shape[0]
        device, dtype = sat.device, sat.dtype

        coarse = torch.deg2rad(torch.arange(
            -180.0, 180.0, self.coarse_step_deg, device=device, dtype=dtype))
        _, L_c, _, _ = _eval_grid(sat, coarse, theta, w)
        k = L_c.argmin(dim=-1)                                           # [Bv]
        psi_c = torch.gather(coarse.unsqueeze(0).expand(Bv, -1),
                             1, k.unsqueeze(1)).squeeze(1)               # [Bv]

        half = np.deg2rad(self.coarse_step_deg)
        offsets = torch.arange(-half, half + 1e-12,
                               np.deg2rad(self.fine_step_deg),
                               device=device, dtype=dtype)               # [Kf]
        fine = psi_c.unsqueeze(1) + offsets.unsqueeze(0)                 # [Bv, Kf]
        C_f, L_f, _, _ = _eval_grid(sat, fine, theta, w)
        kf = L_f.argmin(dim=-1)
        psi = torch.gather(fine, 1, kf.unsqueeze(1)).squeeze(1)          # [Bv]
        C = torch.gather(C_f, 1,
                         kf.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        return C, psi


def solve_ransac(sat_points, theta, weights=None, **solver_kwargs):
    """Functional wrapper around :class:`e2eProbabilisticCrossUnnormSolver`.

    Args:
        sat_points:    [B, N, 2] satellite BEV points (meters).
        theta:         [B, N] ray angles (radians, cw convention).
        weights:       [B, N] matching weights (None -> uniform).
        **solver_kwargs: forwarded to the solver constructor.

    Returns:
        (C_pred, psi_pred, loss, inlier_mask) as in
        :meth:`e2eProbabilisticCrossUnnormSolver.estimate_pose`.
    """
    solver = e2eProbabilisticCrossUnnormSolver(**solver_kwargs)
    return solver.estimate_pose(sat_points, theta, weights)
