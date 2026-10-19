"""Shared geometry / font / image utilities for the figure scripts.

Everything is expressed in the network's own satellite-input frame: the
656x656 input is rotated so the car heading points along +x (screen right),
the car's right side is +y (screen down), and the metric BEV frame
``(mx, my) = (right, forward)`` metres — origin at the input centre — maps
to input pixels through ``x = 328 + my * S``, ``y = 328 + mx * S`` with
``S = 656 / grid_size_h``. No external map tiles or metadata are involved.
"""
import configparser
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from PIL import Image

GEO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRD_W, GRD_H = 1408, 416            # network ground input (px)
FEAT_W = 88                         # ground feature-map width (1408 / 16)
SAT_GRID = 41                       # satellite feature-map width
SAT_INPUT = 656                     # network satellite input (px)
GRID_M = 128.44                     # BEV grid side length (m), config [KITTI]

COLORS = ['#1f77b4', '#ff7f0e', '#9467bd', '#17becf', '#e377c2',
          '#8c564b', '#7f7f7f', '#aec7e8', '#ffbb78', '#c49c94']

_FONT_FILES = ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf')
_FONT_DIRS = (os.path.expanduser('~/.local/share/fonts'),
              '/usr/share/fonts/truetype/dejavu')


def setup_matplotlib_font():
    """Times New Roman with Liberation Serif / DejaVu Serif fallbacks."""
    for d in _FONT_DIRS:
        for f in _FONT_FILES:
            p = os.path.join(d, f)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
    })


def pil_font(size=28, bold=True):
    """PIL truetype font: Times (bold) -> DejaVu Sans (bold) -> default."""
    from PIL import ImageFont
    names = ['timesbd.ttf', 'times.ttf'] if bold else \
        ['times.ttf', 'timesbd.ttf']
    for d in _FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    for d in _FONT_DIRS:
        p = os.path.join(d, 'DejaVuSans-Bold.ttf' if bold
                         else 'DejaVuSans.ttf')
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def to_656(m, grid_m=GRID_M):
    """Metric BEV (right, forward) m -> 656-frame px (x=forward, y=right)."""
    s = SAT_INPUT / grid_m
    return np.array([m[1] * s + SAT_INPUT / 2, m[0] * s + SAT_INPUT / 2])


def pick_gt_points(depth, n=10, seed=0, d_max=30.0):
    """Depth-stratified road points (row > 250 covers road within 35 m).

    [1.5, d_max] is split into n equal bins; one random pixel per bin
    (empty bins back-filled) so points spread from near to far.
    Returns [(u, v, depth_m)] with u/v network-resolution pixels.
    """
    rng = np.random.RandomState(seed)
    rows = np.arange(depth.shape[0])[:, None]
    cand = np.argwhere((depth > 1.5) & (depth < d_max) & (rows > 250))
    keep = (cand[:, 1] > 8) & (cand[:, 1] < GRD_W - 8) & \
           (cand[:, 0] > 30) & (cand[:, 0] < GRD_H - 10)
    cand = cand[keep]
    ds = depth[cand[:, 0], cand[:, 1]]
    bins = np.linspace(1.5, d_max, n + 1)
    out, used = [], set()
    for b0, b1 in zip(bins[:-1], bins[1:]):
        ids = np.where((ds > b0) & (ds <= b1))[0]
        if len(ids):
            i = int(ids[rng.randint(len(ids))])
            out.append(i)
            used.add(i)
    if len(out) < n:
        rest = [i for i in range(len(cand)) if i not in used]
        for i in rng.choice(rest, size=min(n - len(out), len(rest)),
                            replace=False):
            out.append(int(i))
    return [(int(cand[i, 1]), int(cand[i, 0]),
             float(depth[cand[i, 0], cand[i, 1]])) for i in out]


def tensor2img(img_tensor):
    """De-normalised network input tensor [3, H, W] -> PIL RGB image."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    arr = (img * 255.0).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(arr)


def kitti_root():
    """KITTI dataset root from config.ini [KITTI]."""
    config = configparser.ConfigParser()
    config.read(os.path.join(GEO_ROOT, 'config.ini'))
    return config['KITTI']['dataset_root']
