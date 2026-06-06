"""
ComfyUI-SEGA-HYVideo
Spectral-Energy Guided Attention for HunyuanVideo — resolution extrapolation
without retraining, new weights, or architecture changes.

Based on: SEGA (arXiv:2605.22668) — ported from FLUX to HunyuanVideo's 3D RoPE.
Target: Graydient.AI (24 GB VRAM + 100 GB RAM pool)
"""

from .comfyui_sega_hyvideo.nodes import (
    HunyuanVideoSEGAConfig,
    HunyuanVideoSEGAPatch,
)

NODE_CLASS_MAPPINGS = {
    "HunyuanVideoSEGAConfig": HunyuanVideoSEGAConfig,
    "HunyuanVideoSEGAPatch":  HunyuanVideoSEGAPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoSEGAConfig": "SEGA Config (HunyuanVideo)",
    "HunyuanVideoSEGAPatch":  "SEGA Model Patch (HunyuanVideo)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
