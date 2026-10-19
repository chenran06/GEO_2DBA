"""Frozen DINOv3 ViT-L/16 feature extractor (RoMaV2 weights).

The backbone is loaded lazily on first use and shared across training and
evaluation. It is never part of the trainable state: only the matcher
parameters are stored in checkpoints.

Feature extraction returns intermediate layer 11 (after truncating the
network to 12 blocks), reshaped to a dense map with a stride of 16.
"""
import torch
from einops import rearrange

_HUB_COMMIT = "adc254450203739c8149213a7a69d8d905b4fcfa"
_ROMAV2_URL = ("https://github.com/Parskatt/RoMaV2/releases/download/"
               "v2.0.1/romav2.0.1.pt")

_model = None
_device = None


def _load_model(device):
    """Build the ViT-L/16 backbone once and keep it at module level.

    Raises:
        RuntimeError: if the hub repo or the RoMaV2 weights cannot be
            downloaded (offline environments must pre-populate the torch hub
            cache).
    """
    model = torch.hub.load(
        repo_or_dir=f"facebookresearch/dinov3:{_HUB_COMMIT}",
        model="dinov3_vitl16",
        pretrained=False,
        weights=None,
        skip_validation=True,
    ).to(torch.bfloat16).to(device)
    model.eval()

    ckpt = torch.hub.load_state_dict_from_url(_ROMAV2_URL, map_location=device)
    # RoMaV2 stores the backbone under the "f." prefix; strip it.
    model.load_state_dict({k[2:]: v for k, v in ckpt.items() if k.startswith("f.")})

    # Truncate to 12 blocks: intermediate layer 11 becomes the last block.
    del model.blocks[12:24]
    return model


def img_infer(img_n, H, W):
    """Extract dense DINOv3 features for a batch of normalised images.

    Args:
        img_n: [B, 3, H*16, W*16] ImageNet-normalised input images.
        H, W: target feature-map size (input size / 16).
    Returns:
        [B, 1024, H, W] float feature map (bf16 autocast internally).
    """
    global _model, _device
    if _model is None:
        _device = img_n.device
        _model = _load_model(_device)

    with (torch.autocast(_device.type, torch.bfloat16, enabled=True),
          torch.set_grad_enabled(False)):
        trimmed_out = _model.get_intermediate_layers(img_n, n=[11])
        feat = rearrange(trimmed_out[0], "B (H W) D -> B H W D", H=H, W=W)
        feat = rearrange(feat, "B H W D -> B D H W", H=H, W=W)
    return feat
