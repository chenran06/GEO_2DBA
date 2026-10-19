"""Shared Depth-Anything-V2 metric-depth model builder (KITTI + RobotCar).

Requires the Depth-Anything-V2 repository to be cloned under the external
root (see ``preprocess/README.md``) and that root inserted into
``sys.path`` before calling :func:`build_depth_anything`.
"""

import torch

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def build_depth_anything(encoder, max_depth, load_from, device):
    """Instantiate DepthAnythingV2 with the metric (ViT) checkpoint."""
    from depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(load_from, map_location='cpu'))
    return model.to(device).eval()
