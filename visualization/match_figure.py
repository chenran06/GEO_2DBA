"""Match figures: ``match_gt5.png`` (3-panel) and ``match_model.pdf/png``.

``draw_match_gt5`` reconstructs the legacy v1 layout (street | full
satellite input | zoom): 5 depth-stratified GT street points are matched to
the satellite cells the model scores highest (argmax of the matching
probability column), with GT/predicted pose arrows and the error box.

``draw_match_model`` maps 10 model-sampled pairs onto the network's own
656x656 satellite input (car heading along +x, screen right) and draws the
satellite-top / street-bottom match figure with the camera markers.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

from .vis_common import (FEAT_W, GRD_H, GRD_W, SAT_INPUT, COLORS, GRID_M,
                         pil_font, pick_gt_points, setup_matplotlib_font,
                         to_656)

setup_matplotlib_font()

RED = (235, 30, 30)
GREEN = (0, 190, 0)
ORANGE = (255, 140, 0)
YELLOW = (255, 255, 0)

GRD_PAN_H = 512                    # street panel height
LW_LINE = 6                        # match line width
R_GRD, R_SAT, R_ZOOM = 7, 5, 9     # point radius (street / sat / zoom)
L_ARROW, LW_ARROW, R_ARROW = 80, 9, 11
N_PTS = 5
CELL = 16                          # dedup grid (px) for street points
MARGIN = 40                        # zoom bbox margin (656-frame px)
ZOOM_MIN, ZOOM_MAX = 144, 600      # zoom window side clamp (656-frame px)


def select_points(depth, n=N_PTS, seed=42, d_max=35.0):
    """Depth-stratified street points (16 px cell dedup): 1.5-35 m in n bins,
    empty bins back-filled from a 20-point draw. Returns [(u, v, depth_m)]."""
    seen_g, sel = set(), []

    def add(u, v, D):
        g = (v // CELL) * FEAT_W + (u // CELL)
        if g in seen_g:
            return False
        seen_g.add(g)
        sel.append((u, v, D))
        return True

    for u, v, D in pick_gt_points(depth, n=n, seed=seed, d_max=d_max):
        add(u, v, D)
    if len(sel) < n:
        for u, v, D in pick_gt_points(depth, n=20, seed=seed, d_max=d_max):
            if len(sel) == n:
                break
            add(u, v, D)
    return sel


def argmax_sat_px(matching_score, sel, sat_grid=41):
    """Model argmax satellite endpoint (656-frame px) per street point.

    matching_score: [N_sat, N_grd] or [1, N_sat, N_grd] probabilities.
    Street pixel (u, v) addresses ground column g = (v//16)*FEAT_W + u//16;
    the argmax row s indexes the satellite cell (i, j) = divmod(s, sat_grid),
    whose 656-frame centre pixel is (j*16+8, i*16+8).
    """
    score = torch.as_tensor(matching_score).float()
    if score.dim() == 3:
        score = score[0]
    score = score.T                                  # [N_grd, N_sat]
    out = []
    for u, v, _ in sel:
        g = (v // CELL) * FEAT_W + (u // CELL)
        s = int(torch.argmax(score[g]).item())
        i, j = divmod(s, sat_grid)
        out.append((j * 16 + 8, i * 16 + 8))
    return out


def _to_zoom(p, x0, y0, zx, zx0):
    """656-frame px -> zoom panel canvas px."""
    return ((p[0] - x0) * zx + zx0, (p[1] - y0) * zx)


def draw_match_gt5(grd_img, sat_img, sel, sat_px, C_gt, C_pred, psi_pred,
                   c_err, y_err, out_path, psi_gt=0.0):
    """3-panel match figure (street | satellite | zoom), v1 layout.

    Args:
        grd_img:  network ground input PIL (1408x416 or any aspect).
        sat_img:  network satellite input PIL (656x656).
        sel:      [(u, v, depth_m)] street points (original-image px).
        sat_px:   [(x, y)] model argmax endpoints, 656-frame px.
        C_gt / C_pred: metric-frame positions (right, forward) m.
        psi_pred: predicted yaw (rad, clockwise from forward).
        c_err / y_err: loc / yaw error (m / deg) for the error box.
        psi_gt:   GT yaw (rad; 0 for the KITTI known-orientation protocol).
    """
    w_new = int(round(GRD_PAN_H / GRD_H * grd_img.width))
    pan = grd_img.resize((w_new, GRD_PAN_H), Image.LANCZOS)
    sat = sat_img.resize((GRD_PAN_H, GRD_PAN_H), Image.LANCZOS)

    px = np.asarray(sat_px, float)
    lo, hi = px.min(0), px.max(0)
    Wz = int(min(max((hi - lo).max() + 2 * MARGIN, ZOOM_MIN), ZOOM_MAX))
    ctr = (lo + hi) / 2
    x0 = float(np.clip(ctr[0] - Wz / 2, 0, SAT_INPUT - Wz))
    y0 = float(np.clip(ctr[1] - Wz / 2, 0, SAT_INPUT - Wz))
    zx = GRD_PAN_H / Wz

    canvas = Image.new('RGB', (w_new + 2 * GRD_PAN_H, GRD_PAN_H))
    canvas.paste(pan, (0, 0))
    canvas.paste(sat, (w_new, 0))
    zoom = sat_img.crop((int(x0), int(y0), int(x0) + Wz, int(y0) + Wz)) \
        .resize((GRD_PAN_H, GRD_PAN_H), Image.LANCZOS)
    canvas.paste(zoom, (w_new + GRD_PAN_H, 0))
    draw = ImageDraw.Draw(canvas)
    zx0 = w_new + GRD_PAN_H
    s_sat = GRD_PAN_H / SAT_INPUT    # 656-frame -> sat panel scale

    def to_sat(p):
        return (w_new + p[0] * s_sat, p[1] * s_sat)

    # match lines: street -> satellite panel, then satellite -> zoom panel
    for (u, v, _), p in zip(sel, sat_px):
        x1, y1 = u * w_new / GRD_W, v * GRD_PAN_H / GRD_H
        x2, y2 = to_sat(p)
        xz, yz = _to_zoom(p, x0, y0, zx, zx0)
        draw.line([x1, y1, x2, y2], fill=RED, width=LW_LINE)
        draw.line([x2, y2, xz, yz], fill=RED, width=LW_LINE)
    for (u, v, _), p in zip(sel, sat_px):
        x1, y1 = u * w_new / GRD_W, v * GRD_PAN_H / GRD_H
        x2, y2 = to_sat(p)
        xz, yz = _to_zoom(p, x0, y0, zx, zx0)
        draw.ellipse([x1 - R_GRD, y1 - R_GRD, x1 + R_GRD, y1 + R_GRD],
                     fill=RED)
        draw.ellipse([x2 - R_SAT, y2 - R_SAT, x2 + R_SAT, x2 + R_SAT],
                     fill=RED)
        draw.ellipse([xz - R_ZOOM, yz - R_ZOOM, xz + R_ZOOM, yz + R_ZOOM],
                     fill=RED)

    # zoom-window rectangle on the satellite panel
    draw.rectangle([w_new + x0 * s_sat, y0 * s_sat,
                    w_new + (x0 + Wz) * s_sat, (y0 + Wz) * s_sat],
                   outline=YELLOW, width=4)

    # GT / Pred pose arrows (656-frame canvas: +x = forward, +y = right)
    gt656 = to_656(np.asarray(C_gt, float))
    pr656 = to_656(np.asarray(C_pred, float))

    def draw_pose_arrow(p, psi, color, label, on_zoom):
        if on_zoom:
            x, y = _to_zoom(p, x0, y0, zx, zx0)
            scale = zx
        else:
            x, y = to_sat(p)
            scale = s_sat
        L = L_ARROW * scale
        dx, dy = np.cos(psi) * L, np.sin(psi) * L
        draw.line([x, y, x + dx, y + dy], fill=color, width=LW_ARROW)
        draw.ellipse([x - R_ARROW, y - R_ARROW, x + R_ARROW, y + R_ARROW],
                     fill=color)
        draw.text((x + 10, y - 30), label, fill=color, font=pil_font(28))

    draw_pose_arrow(gt656, psi_gt, GREEN, 'GT', on_zoom=False)
    draw_pose_arrow(pr656, float(psi_pred), ORANGE, 'Pred', on_zoom=False)
    draw_pose_arrow(gt656, psi_gt, GREEN, 'GT', on_zoom=True)
    draw_pose_arrow(pr656, float(psi_pred), ORANGE, 'Pred', on_zoom=True)

    # error box on the zoom panel
    font = pil_font(28)
    txt = f"loc err: {c_err:.2f} m   yaw err: {y_err:.2f} deg"
    tw = draw.textlength(txt, font=font)
    draw.rectangle([zx0 + 4, 4, zx0 + tw + 16, 40], fill=(0, 0, 0))
    draw.text((zx0 + 10, 8), txt, fill=(255, 255, 255), font=font)

    canvas.save(out_path)
    return canvas


# ────────────────────────────────────────────────────────────────
# match_model: model pairs on the network satellite input
# ────────────────────────────────────────────────────────────────
DISP_W = 1408                      # panel display width
GAP = 26                           # panel gap (display px)
HEAD_M = 12.0                      # camera heading line length (m)


def draw_match_model(grd_img, sat_img, pairs10, C_gt, C_pred, psi_pred,
                     out_base, grid_m=GRID_M):
    """match_model figure: satellite input (top) | street image (bottom).

    Args:
        grd_img:  1408x416 street PIL (network resolution).
        sat_img:  656x656 network satellite input PIL.
        pairs10:  [((u, v), (x, y))] model-sampled pairs — (u, v) street px,
                  (x, y) satellite-input px (forward->x, right->y frame).
        C_gt / C_pred: metric-frame positions (right, forward) m.
        psi_pred: predicted yaw (rad, clockwise from forward).
        out_base: output path prefix (writes .pdf and .png).
        grid_m:   BEV grid side length (m).
    """
    s = DISP_W / SAT_INPUT
    dh = DISP_W
    grd_disp = grd_img.resize(
        (DISP_W, int(round(GRD_H * DISP_W / grd_img.width))), Image.LANCZOS)
    gh = grd_disp.height
    sat_disp = sat_img.resize((DISP_W, DISP_W), Image.LANCZOS)
    total_h = dh + GAP + gh

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(DISP_W / 300, total_h / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, DISP_W)
    ax.set_ylim(0, total_h)
    ax.axis('off')
    ax.imshow(np.array(sat_disp), extent=(0, DISP_W, 0, dh), origin='upper',
              interpolation='bilinear')
    ax.imshow(np.array(grd_disp),
              extent=(0, DISP_W, dh + GAP, dh + GAP + gh), origin='upper',
              interpolation='bilinear')

    for k, ((u, v), (x, y)) in enumerate(pairs10):
        cf = COLORS[k % len(COLORS)]
        x0, y0 = u * DISP_W / GRD_W, dh + GAP + (gh - v * gh / GRD_H)
        x1, y1 = x * s, dh - y * s
        ax.plot([x0, x1], [y0, y1], color=cf, lw=1.6, zorder=3)
        ax.scatter([x0, x1], [y0, y1], s=42, marker='o', c=[cf, cf],
                   edgecolors='white', linewidths=0.8, zorder=4)

    def cam_marker(C, psi, color):
        gx, gy = to_656(np.asarray(C, float), grid_m)
        L = HEAD_M * SAT_INPUT / grid_m
        dx, dy = L * np.cos(psi), L * np.sin(psi)
        ax.plot([gx * s, (gx + dx) * s], [dh - gy * s, dh - (gy + dy) * s],
                color=color, lw=2.5, zorder=6, solid_capstyle='round')
        ax.scatter([gx * s], [dh - gy * s], s=70, marker='o', c=color,
                   edgecolors='white', linewidths=1.0, zorder=7)

    cam_marker(C_gt, 0.0, 'red')
    cam_marker(C_pred, float(psi_pred), '#ff8c00')

    fig.savefig(out_base + '.pdf', dpi=300, facecolor='white')
    fig.savefig(out_base + '.png', dpi=150, facecolor='white')
    plt.close(fig)
