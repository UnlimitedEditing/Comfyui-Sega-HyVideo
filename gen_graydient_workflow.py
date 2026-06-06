"""
Generate GraydientWorkflow-sega-hyvideo-v1.json

Usage:
    python gen_graydient_workflow.py

Produces a single Graydient backup JSON embedding both workflow formats.
Edit TARGET_* constants below to change the target resolution / frame count.

Node map:
    1  UNETLoader
    2  HunyuanVideoSEGAConfig
    3  HunyuanVideoSEGAPatch
    4  CLIPLoader
    5  CLIPTextEncode  (positive)
    6  CLIPTextEncode  (negative)
    7  EmptyHunyuanLatentVideo
    8  KSampler
    9  VAELoader
    10 VAEDecode
    11 VHS_VideoCombine
"""

import json

# ── Target resolution (what you want to generate) ──────────────────────────
TARGET_HEIGHT = 1080     # pixels — try 1080 or 1296 (1.5× / 1.8× of 720)
TARGET_WIDTH  = 1920     # pixels — try 1920 or 2304
TARGET_FRAMES = 25       # input video frames (25 is the standard HunyuanVideo value)
TARGET_FPS    = 24       # output video frame rate

# ── Training reference (do not change unless you have a different checkpoint) ─
TRAIN_HEIGHT  = 720
TRAIN_WIDTH   = 1280
TRAIN_T_LEN   = 7        # latent temporal length: (25-1)//4 + 1 = 7

# ── Latent dims (VAE 8× spatial, 4× temporal; patch [1,2,2]) ────────────────
LATENT_T = (TARGET_FRAMES - 1) // 4 + 1   # causal VAE: first frame + (rest / 4)

# ──────────────────────────────────────────────────────────────────────────────
# Standard (drag-into-ComfyUI) workflow
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 11,
    "last_link_id": 13,
    "nodes": [
        # 1 — Load UNET weights
        {
            "id": 1, "type": "UNETLoader",
            "pos": [80, 80], "size": {"0": 420, "1": 80},
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "widgets_values": ["hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", "fp8_e4m3fn"],
        },
        # 2 — SEGA config
        {
            "id": 2, "type": "HunyuanVideoSEGAConfig",
            "pos": [80, 220], "size": {"0": 420, "1": 480},
            "inputs": [],
            "outputs": [{"name": "SEGA_HYVIDEO_CONFIG", "type": "SEGA_HYVIDEO_CONFIG", "links": [2], "slot_index": 0}],
            "widgets_values": [
                TRAIN_HEIGHT, TRAIN_WIDTH, TRAIN_T_LEN,   # training reference
                0.15, 1.5, 1.0,                           # mscale_alpha / beta / min
                "power_res", 0.08,                        # formula, coefficient
                0.0, 1.0, 1.5,                            # spread_min / max / alpha
                0.5, True,                                 # temporal_strength, spectral_enabled
            ],
        },
        # 3 — Apply SEGA patch
        {
            "id": 3, "type": "HunyuanVideoSEGAPatch",
            "pos": [600, 120], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "model",       "type": "MODEL",               "link": 1},
                {"name": "sega_config", "type": "SEGA_HYVIDEO_CONFIG", "link": 2},
            ],
            "outputs": [{"name": "model", "type": "MODEL", "links": [3], "slot_index": 0}],
            "widgets_values": [],
        },
        # 4 — Load CLIP
        {
            "id": 4, "type": "CLIPLoader",
            "pos": [80, 760], "size": {"0": 420, "1": 80},
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [4, 5], "slot_index": 0}],
            "widgets_values": ["clip_l.safetensors", "stable_diffusion"],
        },
        # 5 — Positive prompt
        {
            "id": 5, "type": "CLIPTextEncode",
            "pos": [600, 400], "size": {"0": 420, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
            "widgets_values": ["A cinematic shot, ultra high resolution"],
        },
        # 6 — Negative prompt
        {
            "id": 6, "type": "CLIPTextEncode",
            "pos": [600, 600], "size": {"0": 420, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [7], "slot_index": 0}],
            "widgets_values": ["text, watermark, low quality, blurry, distorted"],
        },
        # 7 — Empty latent (resolution + frame count)
        {
            "id": 7, "type": "EmptyHunyuanLatentVideo",
            "pos": [600, 800], "size": {"0": 300, "1": 120},
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [8], "slot_index": 0}],
            "widgets_values": [TARGET_WIDTH, TARGET_HEIGHT, TARGET_FRAMES, 1],
        },
        # 8 — Sampler
        {
            "id": 8, "type": "KSampler",
            "pos": [1060, 300], "size": {"0": 320, "1": 280},
            "inputs": [
                {"name": "model",        "type": "MODEL",        "link": 3},
                {"name": "positive",     "type": "CONDITIONING", "link": 6},
                {"name": "negative",     "type": "CONDITIONING", "link": 7},
                {"name": "latent_image", "type": "LATENT",       "link": 8},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9], "slot_index": 0}],
            # seed, seed_control, steps, cfg, sampler, scheduler, denoise
            "widgets_values": [12345, "randomize", 30, 6.0, "euler", "simple", 1.0],
        },
        # 9 — Load VAE
        {
            "id": 9, "type": "VAELoader",
            "pos": [80, 900], "size": {"0": 420, "1": 80},
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [10], "slot_index": 0}],
            "widgets_values": ["hunyuan_video_vae_bf16.safetensors"],
        },
        # 10 — Decode latent
        {
            "id": 10, "type": "VAEDecode",
            "pos": [1460, 300], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 9},
                {"name": "vae",     "type": "VAE",    "link": 10},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [11], "slot_index": 0}],
            "widgets_values": [],
        },
        # 11 — Combine frames → video (fps controlled here)
        {
            "id": 11, "type": "VHS_VideoCombine",
            "pos": [1860, 300], "size": {"0": 340, "1": 200},
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": 11},
            ],
            "outputs": [],
            # frame_rate, loop_count, filename_prefix, format, pingpong, save_output
            "widgets_values": [TARGET_FPS, 0, "SEGA_HYVideo", "video/h264-mp4", False, True],
        },
    ],
    "links": [
        [1,  1, 0,  3, 0, "MODEL"],
        [2,  2, 0,  3, 1, "SEGA_HYVIDEO_CONFIG"],
        [3,  3, 0,  8, 0, "MODEL"],
        [4,  4, 0,  5, 0, "CLIP"],
        [5,  4, 0,  6, 0, "CLIP"],
        [6,  5, 0,  8, 1, "CONDITIONING"],
        [7,  6, 0,  8, 2, "CONDITIONING"],
        [8,  7, 0,  8, 3, "LATENT"],
        [9,  8, 0, 10, 0, "LATENT"],
        [10, 9, 0, 10, 1, "VAE"],
        [11, 10, 0, 11, 0, "IMAGE"],
    ],
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

# ──────────────────────────────────────────────────────────────────────────────
# API workflow (what Graydient sends to ComfyUI engine)
# ──────────────────────────────────────────────────────────────────────────────
api = {
    "1": {
        "inputs": {
            "unet_name":    "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
            "weight_dtype": "fp8_e4m3fn",
        },
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
        "inputs": {
            "text": "A cinematic shot, ultra high resolution",
            "clip": ["4", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "6": {
        "inputs": {
            "text": "text, watermark, low quality, blurry, distorted",
            "clip": ["4", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "7": {
        "inputs": {
            "width":      TARGET_WIDTH,
            "height":     TARGET_HEIGHT,
            "length":     TARGET_FRAMES,
            "batch_size": 1,
        },
        "class_type": "EmptyHunyuanLatentVideo",
    },
    "8": {
        "inputs": {
            "seed":         12345,
            "steps":        30,
            "cfg":          6.0,
            "sampler_name": "euler",
            "scheduler":    "simple",
            "denoise":      1.0,
            "model":        ["3", 0],
            "positive":     ["5", 0],
            "negative":     ["6", 0],
            "latent_image": ["7", 0],
        },
        "class_type": "KSampler",
    },
    "9": {
        "inputs": {"vae_name": "hunyuan_video_vae_bf16.safetensors"},
        "class_type": "VAELoader",
    },
    "10": {
        "inputs": {"samples": ["8", 0], "vae": ["9", 0]},
        "class_type": "VAEDecode",
    },
    "11": {
        "inputs": {
            "images":          ["10", 0],
            "frame_rate":      TARGET_FPS,
            "loop_count":      0,
            "filename_prefix": "SEGA_HYVideo",
            "format":          "video/h264-mp4",
            "pingpong":        False,
            "save_output":     True,
        },
        "class_type": "VHS_VideoCombine",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Graydient field mappings
# ──────────────────────────────────────────────────────────────────────────────
field_mapping = [
    # ── Prompts ─────────────────────────────────────────────────────────────
    {
        "local_field":      "prompt",
        "node_id":          5,
        "node_input_index": 0,
        "node_input_name":  "text",
        "node_name":        "CLIPTextEncode (positive)",
        "default_value":    "A cinematic shot, ultra high resolution",
        "help_text":        "Positive text prompt",
    },
    {
        "local_field":      "negative_prompt",
        "node_id":          6,
        "node_input_index": 0,
        "node_input_name":  "text",
        "node_name":        "CLIPTextEncode (negative)",
        "default_value":    "text, watermark, low quality, blurry, distorted",
        "help_text":        "Negative text prompt",
    },
    # ── Sampling ────────────────────────────────────────────────────────────
    {
        "local_field":      "seed",
        "node_id":          8,
        "node_input_index": 0,
        "node_input_name":  "seed",
        "node_name":        "KSampler",
        "default_value":    12345,
        "help_text":        "Sampling seed (-1 for random)",
    },
    {
        "local_field":      "steps",
        "node_id":          8,
        "node_input_index": 2,
        "node_input_name":  "steps",
        "node_name":        "KSampler",
        "default_value":    30,
        "help_text":        "Number of denoising steps",
    },
    {
        "local_field":      "guidance",
        "node_id":          8,
        "node_input_index": 3,
        "node_input_name":  "cfg",
        "node_name":        "KSampler",
        "default_value":    6.0,
        "help_text":        "CFG guidance scale",
    },
    {
        "local_field":      "strength",
        "node_id":          8,
        "node_input_index": 6,
        "node_input_name":  "denoise",
        "node_name":        "KSampler",
        "default_value":    1.0,
        "help_text":        "Denoising strength (1.0 = full generation)",
    },
    # ── Video dimensions ────────────────────────────────────────────────────
    {
        "local_field":      "size",
        "node_id":          7,
        "node_input_index": 0,
        "node_input_name":  "width",
        "node_name":        "EmptyHunyuanLatentVideo",
        "default_value":    TARGET_WIDTH,
        "help_text":        "Output video width in pixels",
    },
    {
        "local_field":      "size",
        "node_id":          7,
        "node_input_index": 1,
        "node_input_name":  "height",
        "node_name":        "EmptyHunyuanLatentVideo",
        "default_value":    TARGET_HEIGHT,
        "help_text":        "Output video height in pixels",
    },
    {
        "local_field":      "length",
        "node_id":          7,
        "node_input_index": 2,
        "node_input_name":  "length",
        "node_name":        "EmptyHunyuanLatentVideo",
        "default_value":    TARGET_FRAMES,
        "help_text":        "Number of video frames to generate",
    },
    # ── Output ───────────────────────────────────────────────────────────────
    {
        "local_field":      "fps",
        "node_id":          11,
        "node_input_index": 0,
        "node_input_name":  "frame_rate",
        "node_name":        "VHS_VideoCombine",
        "default_value":    TARGET_FPS,
        "help_text":        "Output video frame rate",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Graydient backup config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            f"HunyuanVideo text-to-video at {TARGET_WIDTH}×{TARGET_HEIGHT} "
            f"({TARGET_FRAMES} frames, {TARGET_FPS} fps) using SEGA 3D spectral-energy "
            f"guided RoPE — training-free resolution extrapolation."
        ),
        "peak_vram_usage": 0,   # fill after first successful run
        "requirements": {
            "github": [
                "https://github.com/UnlimitedEditing/Comfyui-Sega-HyVideo",
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
            ],
            "pip": [],
        },
        "field_mapping": field_mapping,
        "workflow":   json.dumps(standard, indent=2),
        "workflow_2": json.dumps(api,      indent=2),
        "supports_img2img":       False,
        "supports_txt2img":       True,
        "install_detected_nodes": True,
        "is_public":              False,
    }
}

out_path = "GraydientWorkflow-sega-hyvideo-v1.json"
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written : {out_path}")
print(f"Target  : {TARGET_WIDTH}×{TARGET_HEIGHT} @ {TARGET_FRAMES} frames / {TARGET_FPS} fps")
print(f"Training: {TRAIN_WIDTH}×{TRAIN_HEIGHT}   t_len={TRAIN_T_LEN}")
print(f"Scale   : {TARGET_WIDTH/TRAIN_WIDTH:.2f}× spatial  |  "
      f"{LATENT_T}/{TRAIN_T_LEN} temporal latent frames")
