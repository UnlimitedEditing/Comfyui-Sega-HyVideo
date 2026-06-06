"""
Generate GraydientWorkflow-sega-hyvideo-v1.json

Usage:
    python gen_graydient_workflow.py

Produces a single Graydient backup JSON embedding both workflow formats.
Edit TARGET_* constants below to change the target resolution / frame count.
"""

import json

# ── Target resolution (what you want to generate) ──────────────────────────
TARGET_HEIGHT = 1080     # pixels — try 1080 or 1296 (1.5× / 1.8× of 720)
TARGET_WIDTH  = 1920     # pixels — try 1920 or 2304
TARGET_FRAMES = 25       # input video frames (25 is the standard HunyuanVideo value)

# ── Training reference (do not change unless you have a different checkpoint) ─
TRAIN_HEIGHT  = 720
TRAIN_WIDTH   = 1280
TRAIN_T_LEN   = 7        # latent temporal length: (25-1)//4 + 1 = 7

# ── Latent dims (VAE 8× spatial, 4× temporal; patch [1,2,2]) ────────────────
LATENT_H = TARGET_HEIGHT // 8
LATENT_W = TARGET_WIDTH  // 8
# HunyuanVideo causal VAE: first frame is key frame, rest are compressed 4×
LATENT_T = (TARGET_FRAMES - 1) // 4 + 1

# ──────────────────────────────────────────────────────────────────────────────
# Standard (drag-into-ComfyUI) workflow
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 10,
    "last_link_id": 12,
    "nodes": [
        {
            "id": 1, "type": "UNETLoader",
            "pos": [100, 100], "size": {"0": 400, "1": 80},
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "widgets_values": ["hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", "fp8_e4m3fn"],
        },
        {
            "id": 2, "type": "HunyuanVideoSEGAConfig",
            "pos": [100, 230], "size": {"0": 400, "1": 480},
            "inputs": [],
            "outputs": [{"name": "SEGA_HYVIDEO_CONFIG", "type": "SEGA_HYVIDEO_CONFIG", "links": [2], "slot_index": 0}],
            "widgets_values": [
                TRAIN_HEIGHT, TRAIN_WIDTH, TRAIN_T_LEN,   # training reference
                0.15, 1.5, 1.0,                           # mscale_alpha/beta/min
                "power_res", 0.08,                        # formula, coefficient
                0.0, 1.0, 1.5,                            # spread_min/max/alpha
                0.5, True,                                 # temporal_strength, spectral_enabled
            ],
        },
        {
            "id": 3, "type": "HunyuanVideoSEGAPatch",
            "pos": [600, 150], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "model",       "type": "MODEL",               "link": 1},
                {"name": "sega_config", "type": "SEGA_HYVIDEO_CONFIG", "link": 2},
            ],
            "outputs": [{"name": "model", "type": "MODEL", "links": [3], "slot_index": 0}],
            "widgets_values": [],
        },
        {
            "id": 4, "type": "CLIPLoader",
            "pos": [100, 760], "size": {"0": 400, "1": 80},
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [4], "slot_index": 0}],
            "widgets_values": ["clip_l.safetensors", "stable_diffusion"],
        },
        {
            "id": 5, "type": "CLIPTextEncode",
            "pos": [600, 500], "size": {"0": 400, "1": 200},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5], "slot_index": 0}],
            "widgets_values": ["A cinematic shot, ultra high resolution"],
        },
        {
            "id": 6, "type": "EmptyHunyuanLatentVideo",
            "pos": [600, 760], "size": {"0": 300, "1": 120},
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6], "slot_index": 0}],
            "widgets_values": [TARGET_WIDTH, TARGET_HEIGHT, TARGET_FRAMES, 1],
        },
        {
            "id": 7, "type": "KSampler",
            "pos": [1000, 300], "size": {"0": 300, "1": 280},
            "inputs": [
                {"name": "model",        "type": "MODEL",       "link": 3},
                {"name": "positive",     "type": "CONDITIONING","link": 5},
                {"name": "negative",     "type": "CONDITIONING","link": 5},
                {"name": "latent_image", "type": "LATENT",      "link": 6},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7], "slot_index": 0}],
            "widgets_values": [12345, "randomize", 30, 6.0, "euler", "simple", 1.0],
        },
        {
            "id": 8, "type": "VAELoader",
            "pos": [100, 900], "size": {"0": 400, "1": 80},
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [8], "slot_index": 0}],
            "widgets_values": ["hunyuan_video_vae_bf16.safetensors"],
        },
        {
            "id": 9, "type": "VAEDecode",
            "pos": [1400, 300], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 7},
                {"name": "vae",     "type": "VAE",    "link": 8},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}],
            "widgets_values": [],
        },
        {
            "id": 10, "type": "SaveImage",
            "pos": [1800, 300], "size": {"0": 300, "1": 120},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
            "outputs": [],
            "widgets_values": ["SEGA_HYVideo_Output"],
        },
    ],
    "links": [
        [1,  1, 0, 3, 0, "MODEL"],
        [2,  2, 0, 3, 1, "SEGA_HYVIDEO_CONFIG"],
        [3,  3, 0, 7, 0, "MODEL"],
        [4,  4, 0, 5, 0, "CLIP"],
        [5,  5, 0, 7, 1, "CONDITIONING"],
        [6,  6, 0, 7, 3, "LATENT"],
        [7,  7, 0, 9, 0, "LATENT"],
        [8,  8, 0, 9, 1, "VAE"],
        [9,  9, 0, 10, 0, "IMAGE"],
    ],
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

# ──────────────────────────────────────────────────────────────────────────────
# API workflow (what Graydient sends to ComfyUI engine)
# ──────────────────────────────────────────────────────────────────────────────
api = {
    "1": {
        "inputs": {"unet_name": "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"},
        "class_type": "UNETLoader",
    },
    "2": {
        "inputs": {
            "training_height":          TRAIN_HEIGHT,
            "training_width":           TRAIN_WIDTH,
            "training_t_len":           TRAIN_T_LEN,
            "mscale_alpha":             0.15,
            "mscale_beta":              1.5,
            "mscale_min":               1.0,
            "base_mscale_formula":      "power_res",
            "base_mscale_coefficient":  0.08,
            "spread_min":               0.0,
            "spread_max":               1.0,
            "spread_alpha":             1.5,
            "temporal_sega_strength":   0.5,
            "spectral_enabled":         True,
        },
        "class_type": "HunyuanVideoSEGAConfig",
    },
    "3": {
        "inputs": {"model": ["1", 0], "sega_config": ["2", 0]},
        "class_type": "HunyuanVideoSEGAPatch",
    },
    "4": {
        "inputs": {"clip_name": "clip_l.safetensors", "type": "stable_diffusion"},
        "class_type": "CLIPLoader",
    },
    "5": {
        "inputs": {"text": "A cinematic shot, ultra high resolution", "clip": ["4", 0]},
        "class_type": "CLIPTextEncode",
    },
    "6": {
        "inputs": {
            "width":  TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "length": TARGET_FRAMES,
            "batch_size": 1,
        },
        "class_type": "EmptyHunyuanLatentVideo",
    },
    "7": {
        "inputs": {
            "seed": 12345, "steps": 30, "cfg": 6.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model":        ["3", 0],
            "positive":     ["5", 0],
            "negative":     ["5", 0],
            "latent_image": ["6", 0],
        },
        "class_type": "KSampler",
    },
    "8": {
        "inputs": {"vae_name": "hunyuan_video_vae_bf16.safetensors"},
        "class_type": "VAELoader",
    },
    "9": {
        "inputs": {"samples": ["7", 0], "vae": ["8", 0]},
        "class_type": "VAEDecode",
    },
    "10": {
        "inputs": {"filename_prefix": "SEGA_HYVideo_Output", "images": ["9", 0]},
        "class_type": "SaveImage",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Graydient backup config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            f"HunyuanVideo text-to-video at {TARGET_WIDTH}×{TARGET_HEIGHT} ({TARGET_FRAMES} frames) "
            f"using SEGA 3D spectral-energy guided RoPE — training-free resolution extrapolation."
        ),
        "peak_vram_usage": 0,   # fill after first successful run
        "requirements": {
            "github": ["https://github.com/UnlimitedEditing/Comfyui-Sega-HyVideo"],
            "pip":    [],
        },
        "field_mapping": [
            {
                "local_field":       "prompt",
                "node_id":           5,
                "node_input_index":  0,
                "node_input_name":   "text",
                "node_name":         "CLIPTextEncode",
                "default_value":     "A cinematic shot, ultra high resolution",
                "help_text":         "Text prompt for the video",
            },
        ],
        "workflow":   json.dumps(standard, indent=2),
        "workflow_2": json.dumps(api,      indent=2),
        "supports_img2img":         False,
        "supports_txt2img":         True,
        "install_detected_nodes":   True,
        "is_public":                False,
    }
}

out_path = "GraydientWorkflow-sega-hyvideo-v1.json"
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written: {out_path}")
print(f"Target:   {TARGET_WIDTH}×{TARGET_HEIGHT} @ {TARGET_FRAMES} frames")
print(f"Training: {TRAIN_WIDTH}×{TRAIN_HEIGHT}   t_len={TRAIN_T_LEN}")
print(f"Scale:    {TARGET_WIDTH/TRAIN_WIDTH:.2f}× (spatial)  "
      f"{LATENT_T}/{TRAIN_T_LEN} (temporal latent frames)")
