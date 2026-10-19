"""Solver 4-panel figure (solver.png / solver_white.png).

Re-expresses the legacy two-stage solve (``uilt/draw_solver_iteration_clean.py``
+ ``vis_paper/draw_unified_tile.py``) on top of the unified
:mod:`solver.cross_unnorm` module, drawn in the network's satellite-input
frame (heading = +x, car right = +y; no external map tiles needed):

  (a) matched satellite points        (b) coarse scan -> soft argmin
  (c) fine scan -> output             (d) GT vs prediction

``solver_white.png`` is the same figure without the satellite background.
Per-sample panels are also saved individually (``solver_{a,b,c,d}.png``).
"""
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from solver.cross_unnorm import (compute_theta_pinhole, _normalize_weights,
                                 _eval_grid)
from .vis_common import (FEAT_W, GRID_M, SAT_INPUT, setup_matplotlib_font)

setup_matplotlib_font()

# Stage constants of the legacy visualisation (not the training defaults).
COARSE_STEP_DEG = 10.0
CHEIR_LAMBDA = 1.0
TAU_C = 0.05
TAU_F = 0.005
FINE_PTS = 512

BLUE = '#1F77B4'
RED = '#D62728'
GREEN = '#2CA02C'
GRAY = '#606060'


def circ_deg(a, b):
    """Smallest angular distance between two angles in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def solve_stages(sat_b, grd_b, K_b, w_b, feat_w=FEAT_W):
    """Two-stage soft-argmin solve with per-stage internals for drawing.

    Numerically identical to the legacy ``solve_stages``: the unified
    solver's bridged pinhole theta reproduces the same C(psi), losses and
    cheirality at every psi (see ``solver/cross_unnorm.py``).

    Args:
        sat_b:  [1, N, 2] satellite BEV points (m).
        grd_b:  [1, N, 2] ground feature-cell UV.
        K_b:    [1, 3, 3] intrinsics in feature-map pixels.
        w_b:    [1, N] matching weights (None -> uniform).

    Returns:
        dict with coarse_deg/alpha/C_grid_c/psi_coarse/C_coarse and
        fine_psi/beta/C_grid_f/psi_pred/C_pred (all numpy / float).
    """
    theta = compute_theta_pinhole(grd_b, K_b, feat_w)
    w = _normalize_weights(w_b)
    if w is None:
        w = torch.ones_like(theta) / theta.shape[1]

    # Stage 1: full-circle coarse scan + cheirality weighting.
    coarse_deg = torch.arange(-180.0, 180.0, COARSE_STEP_DEG,
                              device=sat_b.device, dtype=sat_b.dtype)
    coarse_rad = torch.deg2rad(coarse_deg)
    C_grid_c, losses_c, cheir_c, _ = _eval_grid(sat_b, coarse_rad, theta, w)
    a = torch.softmax(-(losses_c + CHEIR_LAMBDA * cheir_c) / TAU_C, dim=1)
    psi_coarse = (a * coarse_rad.view(1, -1)).sum(dim=1)
    C_coarse = (a.unsqueeze(-1) * C_grid_c).sum(dim=1)

    # Stage 2: fine scan (pure L) around the coarse soft-argmin centre.
    half = math.radians(COARSE_STEP_DEG)
    offsets = torch.linspace(-half, half, FINE_PTS, device=sat_b.device,
                             dtype=sat_b.dtype)
    fine_psi = psi_coarse.unsqueeze(1) + offsets.unsqueeze(0)
    C_grid_f, losses_f, _, _ = _eval_grid(sat_b, fine_psi, theta, w)
    beta = torch.softmax(-losses_f / TAU_F, dim=1)
    psi_pred = (beta * fine_psi).sum(dim=1)
    C_pred = (beta.unsqueeze(-1) * C_grid_f).sum(dim=1)

    return dict(
        coarse_deg=coarse_deg.cpu().numpy(),
        alpha=a[0].cpu().numpy(),
        C_grid_c=C_grid_c[0].cpu().numpy(),
        psi_coarse=float(psi_coarse[0]),
        C_coarse=C_coarse[0].cpu().numpy(),
        fine_psi=fine_psi[0].cpu().numpy(),
        beta=beta[0].cpu().numpy(),
        C_grid_f=C_grid_f[0].cpu().numpy(),
        psi_pred=float(psi_pred[0]),
        C_pred=C_pred[0].cpu().numpy(),
    )


class InputMapper:
    """Solver metric frame (mx, my) = (right, forward) m -> input-frame px.

    The metric BEV grid is defined on the 656x656 satellite input, so the
    mapping is exact by construction: origin = input centre.
    """

    def __init__(self, grid_m=GRID_M):
        self.grid_m = grid_m
        self.mpp = grid_m / SAT_INPUT        # metres per input pixel
        self.c = SAT_INPUT / 2.0

    def __call__(self, mx, my):
        return np.array([self.c + my / self.mpp, self.c + mx / self.mpp])

    def map_many(self, pts):
        """Batch (N, 2) metric points -> (N, 2) input px."""
        return np.array([self(mx, my) for mx, my in np.asarray(pts, float)])


def square_crop(px_lists, mpp, margin_m=8.0, min_half_m=20.0,
                max_half_px=SAT_INPUT / 2):
    """Square crop window (centre + half size, input px) around point sets."""
    pts = np.vstack([p for p in px_lists if len(p)])
    ccx, ccy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + margin_m / mpp
    half = float(np.clip(half, min_half_m / mpp, max_half_px))
    ccx = float(np.clip(ccx, half, SAT_INPUT - half))
    ccy = float(np.clip(ccy, half, SAT_INPUT - half))
    return ccx, ccy, half


def draw_pose(ax, C, psi_rad, theta_draw, ray_len, color, lw_ray,
              alpha_ray, lw_head, head_len, z=3, ls='-'):
    """One pose: FOV ray fan + central heading bar (input-frame axes, m).

    psi_rad: solver yaw (rad, clockwise from forward); a ray at column
    angle theta points along (cos(psi+theta), -sin(psi+theta)) in the
    plotted axes (screen up = car left).
    """
    for th in theta_draw:
        a = psi_rad + th
        ax.plot([C[0], C[0] + ray_len * np.cos(a)],
                [C[1], C[1] - ray_len * np.sin(a)],
                color=color, lw=lw_ray, alpha=alpha_ray, zorder=z, ls=ls)
    ax.plot([C[0], C[0] + head_len * np.cos(psi_rad)],
            [C[1], C[1] - head_len * np.sin(psi_rad)],
            color=color, lw=lw_head, alpha=1.0, zorder=z + 1,
            solid_capstyle='round', ls=ls)


def draw_candidates(ax, C_cand, psi_cand_rad, wt, theta_draw, ray_len,
                    head_len, z=2):
    """Candidate poses: gray rays + dots with area proportional to weight."""
    wmax = max(wt.max(), 1e-12)
    for i in range(len(C_cand)):
        draw_pose(ax, C_cand[i], psi_cand_rad[i], theta_draw, ray_len,
                  GRAY, lw_ray=0.6, alpha_ray=0.65, lw_head=1.3,
                  head_len=head_len, z=z)
        s = 25.0 + 260.0 * (wt[i] / wmax)
        ax.scatter(C_cand[i, 0], C_cand[i, 1], s=s, c=GRAY,
                   alpha=0.70, edgecolors='#333333', linewidths=0.5,
                   zorder=z + 2)


def draw_solver(mapper, st, sat_np, C_gt, c_err_ref, y_err_ref, camera_k,
                sat_img, out_dir, grid_m=GRID_M, bg_white=False):
    """Solver 4-panel figure with the satellite-input crop as background.

    Args:
        mapper:    :class:`InputMapper` (metric frame -> input px).
        st:        :func:`solve_stages` output dict.
        sat_np:    (N, 2) metric-frame matched satellite points.
        C_gt:      metric-frame GT position.
        c_err_ref / y_err_ref: reference errors (consistency check).
        camera_k:  intrinsics in feature-map pixels (for the ray fan).
        sat_img:   656x656 network satellite input PIL image (background).
        out_dir:   output directory.
        grid_m:    BEV grid side length (m); must match the solver frame.
        bg_white:  True -> white background variant (``solver_white.*``).

    Returns:
        info dict (gt_px / pred_px / psi_pred_deg / c_err / y_err).
    """
    mpp = mapper.mpp
    gt_px = mapper(*C_gt)
    pred_px = mapper(*st['C_pred'])
    coarse_px = mapper(*st['C_coarse'])
    sat_px = mapper.map_many(sat_np)

    ic = int(np.argmax(st['alpha']))
    Kc = len(st['coarse_deg'])
    c_idx = [(ic - 1) % Kc, ic, (ic + 1) % Kc]
    cand_c_px = mapper.map_many(st['C_grid_c'][c_idx])
    a_cand = st['alpha'][c_idx]
    jf = int(np.argmax(st['beta']))
    step = FINE_PTS // 3
    f_idx = [max(jf - step, 0), jf, min(jf + step, FINE_PTS - 1)]
    cand_f_px = mapper.map_many(st['C_grid_f'][f_idx])
    b_cand = st['beta'][f_idx]

    ccx, ccy, half = square_crop([sat_px, gt_px[None], pred_px[None],
                                  coarse_px[None], cand_c_px, cand_f_px],
                                 mpp)
    Hm = half * mpp
    crop = sat_img.crop((int(round(ccx - half)), int(round(ccy - half)),
                         int(round(ccx + half)), int(round(ccy + half))))

    def xy(p):
        return np.array([(p[0] - ccx) * mpp, -(p[1] - ccy) * mpp])

    gt_xy, pred_xy, coarse_xy = xy(gt_px), xy(pred_px), xy(coarse_px)
    pts_xy = np.array([xy(p) for p in sat_px])
    cand_c_xy = np.array([xy(p) for p in cand_c_px])
    cand_f_xy = np.array([xy(p) for p in cand_f_px])

    fx = float(camera_k[0, 0])
    cxk = float(camera_k[0, 2])
    u = np.linspace(0, FEAT_W - 1, 12)
    theta_draw = np.arctan((u - cxk) / max(fx, 1e-8))

    psi_coarse = st['psi_coarse']
    psi_pred = st['psi_pred']
    cand_c_psi = np.radians(st['coarse_deg'][c_idx])
    cand_f_psi = st['fine_psi'][f_idx]
    c_err = float(np.linalg.norm(st['C_pred'] - C_gt))
    y_err = circ_deg(np.degrees(psi_pred), 0.0)
    if abs(c_err - c_err_ref) > 0.05 or abs(y_err - y_err_ref) > 0.05:
        print(f'    [warn] recomputed err ({c_err:.3f}, {y_err:.3f}) != '
              f'reference ({c_err_ref:.3f}, {y_err_ref:.3f})')

    d_cand = np.linalg.norm(np.vstack([cand_c_xy, cand_f_xy, coarse_xy,
                                       pred_xy]) - gt_xy, axis=1)
    d_pts = np.linalg.norm(pts_xy - gt_xy, axis=1)
    zhalf = float(np.clip(max(d_cand.max() + 4.0, np.percentile(d_pts, 12)),
                          8.0, 15.0))
    gx = float(np.clip(gt_xy[0], -Hm + zhalf, Hm - zhalf))
    gy = float(np.clip(gt_xy[1], -Hm + zhalf, Hm - zhalf))
    ray_len = float(np.clip(0.55 * 2 * Hm * np.sqrt(2), 20, 100))
    head_len = max(4.0, 0.13 * ray_len)
    ray_len_z = 3.0 * zhalf
    head_len_z = max(2.5, 0.30 * zhalf)

    DPI, PANEL, M, GAP, TITLE = 200, 512, 4, 30, 64
    fw = M + PANEL + GAP + PANEL + M
    fh = TITLE + PANEL + GAP + PANEL + M
    fig = plt.figure(figsize=(fw / DPI, fh / DPI), dpi=DPI)
    pw, ph = PANEL / fw, PANEL / fh
    ax1 = fig.add_axes([M / fw, (M + PANEL + GAP) / fh, pw, ph])
    ax2 = fig.add_axes([(M + PANEL + GAP) / fw, (M + PANEL + GAP) / fh,
                        pw, ph])
    ax3 = fig.add_axes([M / fw, M / fh, pw, ph])
    ax4 = fig.add_axes([(M + PANEL + GAP) / fw, M / fh, pw, ph])

    def setup(ax, title, xlim, ylim):
        cx, cy = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
        hs = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        ax.set_xlim(cx - hs, cx + hs)
        ax.set_ylim(cy - hs, cy + hs)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_axis_off()

    def background(ax):
        if bg_white:
            return
        ax.imshow(np.asarray(crop), extent=(-Hm, Hm, -Hm, Hm),
                  origin='upper', zorder=0, interpolation='bilinear')

    full = (-Hm, Hm)

    # (a) matched satellite points
    background(ax1)
    ax1.scatter(pts_xy[:, 0], pts_xy[:, 1], s=8, c=BLUE, alpha=1.0,
                edgecolors='white', linewidths=0.3, zorder=3)
    setup(ax1, f"(a) Matched satellite points (N={len(pts_xy)})",
          full, full)

    # (b) coarse scan
    background(ax2)
    in_win = (np.abs(pts_xy[:, 0] - gx) <= zhalf) & \
             (np.abs(pts_xy[:, 1] - gy) <= zhalf)
    pin = pts_xy[in_win]
    if len(pin):
        ax2.scatter(pin[:, 0], pin[:, 1], s=24, c=BLUE, alpha=0.85,
                    edgecolors='none', zorder=3)
    draw_candidates(ax2, cand_c_xy, cand_c_psi,
                    a_cand, theta_draw, ray_len_z, ray_len_z)
    draw_pose(ax2, coarse_xy, psi_coarse, theta_draw, ray_len_z, RED,
              lw_ray=0.5, alpha_ray=0.9, lw_head=1.8, head_len=ray_len_z, z=5)
    ax2.scatter(coarse_xy[0], coarse_xy[1], s=70, c=RED,
                edgecolors='white', linewidths=0.8, zorder=8)
    ax2.annotate(f"$\\psi_{{coarse}}$={np.degrees(psi_coarse):.1f}°"
                 .replace("-0.0", "0.0"),
                 xy=(coarse_xy[0], coarse_xy[1]), xytext=(-10, 4),
                 textcoords='offset points', ha='right', fontsize=6.5,
                 color=RED)
    setup(ax2, "(b) Coarse scan (10° grid) → soft argmin",
          (gx - zhalf, gx + zhalf), (gy - zhalf, gy + zhalf))
    ax2.legend(handles=[
        plt.Line2D([0], [0], color=GRAY, lw=1.4,
                   label='coarse candidates (size $\\propto \\alpha$)'),
        plt.Line2D([0], [0], color=RED, lw=2.0,
                   label='soft argmin (coarse)'),
    ], loc='upper left', framealpha=0.85, edgecolor='#CCCCCC')

    # (c) fine scan
    background(ax3)
    if len(pin):
        ax3.scatter(pin[:, 0], pin[:, 1], s=24, c=BLUE, alpha=0.85,
                    edgecolors='none', zorder=3)
    draw_candidates(ax3, cand_f_xy, cand_f_psi,
                    b_cand, theta_draw, ray_len_z, ray_len_z)
    draw_pose(ax3, pred_xy, psi_pred, theta_draw, ray_len_z, RED,
              lw_ray=0.5, alpha_ray=0.9, lw_head=1.8, head_len=ray_len_z, z=5)
    ax3.scatter(pred_xy[0], pred_xy[1], s=70, c=RED,
                edgecolors='white', linewidths=0.8, zorder=8)
    ax3.annotate(f"$\\psi_{{pred}}$={np.degrees(psi_pred):.1f}°"
                 .replace("-0.0", "0.0"),
                 xy=(pred_xy[0], pred_xy[1]), xytext=(-10, 4),
                 textcoords='offset points', ha='right', fontsize=6.5,
                 color=RED)
    setup(ax3, "(c) Fine scan (±10°, 512 pts) → output",
          (gx - zhalf, gx + zhalf), (gy - zhalf, gy + zhalf))
    ax3.legend(handles=[
        plt.Line2D([0], [0], color=GRAY, lw=1.4,
                   label='fine candidates (size $\\propto \\beta$)'),
        plt.Line2D([0], [0], color=RED, lw=2.0,
                   label='output (soft argmin)'),
    ], loc='upper left', framealpha=0.85, edgecolor='#CCCCCC')

    # (d) GT vs pred
    background(ax4)
    ax4.scatter(pts_xy[:, 0], pts_xy[:, 1], s=8, c=BLUE, alpha=0.75,
                edgecolors='none', zorder=2)
    draw_pose(ax4, gt_xy, 0.0, theta_draw, ray_len, GREEN,
              lw_ray=0.9, alpha_ray=0.75, lw_head=2.4, head_len=ray_len,
              z=4, ls='--')
    ax4.scatter(gt_xy[0], gt_xy[1], s=60, c=GREEN, edgecolors='white',
                linewidths=0.6, zorder=7)
    draw_pose(ax4, pred_xy, psi_pred, theta_draw, ray_len, RED,
              lw_ray=0.45, alpha_ray=0.65, lw_head=1.6, head_len=ray_len,
              z=5)
    ax4.scatter(pred_xy[0], pred_xy[1], s=60, c=RED, edgecolors='white',
                linewidths=0.6, zorder=8)
    ax4.annotate('GT', xy=(gt_xy[0], gt_xy[1]), xytext=(-10, 12),
                 textcoords='offset points', fontsize=6.5, color=GREEN,
                 ha='right', fontweight='bold')
    ax4.annotate('pred', xy=(pred_xy[0], pred_xy[1]), xytext=(10, -14),
                 textcoords='offset points', fontsize=6.5, color=RED,
                 ha='left', fontweight='bold')
    setup(ax4, f"(d) GT vs pred (C {c_err:.2f} m, yaw {y_err:.2f}°)",
          full, full)
    ax4.legend(handles=[
        plt.Line2D([0], [0], color=GREEN, lw=2.0, label='GT'),
        plt.Line2D([0], [0], color=RED, lw=2.0, label='prediction'),
    ], loc='upper left', framealpha=0.85, edgecolor='#CCCCCC')

    fig.suptitle("2D Bundle Adjustment", fontsize=9.5, y=1 - 2 / fh)
    base = 'solver_white' if bg_white else 'solver'
    fig.savefig(os.path.join(out_dir, base + '.png'), dpi=200,
                facecolor='white')
    fig.savefig(os.path.join(out_dir, base + '.pdf'), dpi=300,
                facecolor='white')
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    hpx = buf.shape[0]
    for ax, tag in ((ax1, 'a'), (ax2, 'b'), (ax3, 'c'), (ax4, 'd')):
        ext = ax.get_window_extent()
        xa, xb = int(round(ext.x0)), int(round(ext.x1))
        ya, yb = int(round(ext.y0)), int(round(ext.y1))
        sub = buf[hpx - yb:hpx - ya, xa:xb]
        Image.fromarray(sub[..., :3]).save(
            os.path.join(out_dir, f'{base}_{tag}.png'))
    plt.close(fig)
    return dict(gt_px=gt_px, pred_px=pred_px,
                psi_pred_deg=float(np.degrees(psi_pred)),
                c_err=c_err, y_err=y_err)
