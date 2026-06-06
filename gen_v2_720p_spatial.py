"""
Generate GraydientWorkflow-sega-hyvideo-v2-720p-spatial.json

Workflow 2: FastHunyuan 720p GGUF + SEGA spatial extrapolation
  - Spatial:  1280×720 → 1920×1088  (1.5×W × 1.511×H)
  - Temporal: 25 frames, no extension  (temporal_sega_strength=0)

HEIGHT MUST BE A MULTIPLE OF 16:  height / 8 (VAE) / 2 (patch_size) must be integer.
  1080 / 16 = 67.5 — INVALID (causes img reshape crash in model.py)
  1088 / 16 = 68   — VALID  (68 patch rows, ~1.511× scale — essentially 1.5×)
  1072 / 16 = 67   — VALID  (67 patch rows, ~1.489× scale)

This is the safer first test — original HunyuanVideo architecture (FastVideo
distillation), confirmed working class_types from Captain's boringvideo workflow.

SEGA spatial NTK:   s_w = 120/80 = 1.5,   d_sp = 56  →  ntk_sp ≈ 2.1
                    s_h = 68/45  = 1.511,  d_sp = 56  →  ntk_sp ≈ 2.1 (same)

NOTE: GGUF nodes require ComfyUI-GGUF custom node.
      Added to github requirements automatically.

Node map:
    1  UnetLoaderGGUF              (FastHunyuan 720p Q8_0)
    2  HunyuanVideoSEGAConfig
    3  HunyuanVideoSEGAPatch
    4  DualCLIPLoaderGGUF          (clip_l + llava-llama GGUF)
    5  CLIPTextEncode  (positive)
    6  CLIPTextEncode  (negative)
    7  EmptyHunyuanLatentVideo
    8  KSampler                    (30 steps, euler, sgm_uniform, cfg 6)
    9  VAELoader
    10 VAEDecodeTiled               (critical at 1088p)
    11 VHS_VideoCombine
"""

import json

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_HEIGHT = 1088   # 1088 = 68×16 — HunyuanVideo requires height % 16 == 0
TARGET_WIDTH  = 1920   # 1920 = 120×16 — valid
TARGET_FRAMES = 25
TARGET_FPS    = 24

# ── Training reference for SEGA ────────────────────────────────────────────
# HunyuanVideo 720p defaults: 1280×720, 25 frames
# Latent t_len = (25-1)//4 + 1 = 7
TRAIN_HEIGHT  = 720
TRAIN_WIDTH   = 1280
TRAIN_T_LEN   = 7

# HunyuanVideo constraint: height & width must be multiples of 16
# (8× VAE spatial compression × 2 patch stride — both must divide evenly)
# 1080 % 16 == 8 → INVALID, causes img reshape crash in forward_orig
assert TARGET_HEIGHT % 16 == 0, f"TARGET_HEIGHT {TARGET_HEIGHT} must be divisible by 16"
assert TARGET_WIDTH  % 16 == 0, f"TARGET_WIDTH {TARGET_WIDTH} must be divisible by 16"
assert TRAIN_HEIGHT  % 16 == 0, f"TRAIN_HEIGHT {TRAIN_HEIGHT} must be divisible by 16"
assert TRAIN_WIDTH   % 16 == 0, f"TRAIN_WIDTH {TRAIN_WIDTH} must be divisible by 16"

# ──────────────────────────────────────────────────────────────────────────────
# Standard workflow
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 11,
    "last_link_id": 12,
    "nodes": [
        # 1 — Load GGUF diffusion model
        {
            "id": 1, "type": "UnetLoaderGGUF",
            "pos": [80, 80], "size": {"0": 460, "1": 80},
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "widgets_values": [
                "fast-hunyuan-video-t2v-720p-Q8_0.gguf",
            ],
        },
        # 2 — SEGA config (spatial extension only, temporal off)
        {
            "id": 2, "type": "HunyuanVideoSEGAConfig",
            "pos": [80, 210], "size": {"0": 460, "1": 480},
            "inputs": [],
            "outputs": [{"name": "SEGA_HYVIDEO_CONFIG", "type": "SEGA_HYVIDEO_CONFIG", "links": [2], "slot_index": 0}],
            "widgets_values": [
                TRAIN_HEIGHT, TRAIN_WIDTH, TRAIN_T_LEN,
                0.15, 1.5, 1.0,        # mscale_alpha / beta / min
                "power_res", 0.08,     # formula, coefficient
                0.0, 1.0, 1.5,         # spread_min / max / alpha
                0.0, True,              # temporal_sega_strength=0 (spatial only), spectral=True
            ],
        },
        # 3 — Apply SEGA patch
        {
            "id": 3, "type": "HunyuanVideoSEGAPatch",
            "pos": [640, 160], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "model",       "type": "MODEL",               "link": 1},
                {"name": "sega_config", "type": "SEGA_HYVIDEO_CONFIG", "link": 2},
            ],
            "outputs": [{"name": "model", "type": "MODEL", "links": [3], "slot_index": 0}],
            "widgets_values": [],
        },
        # 4 — GGUF Dual CLIP loader (clip_l safetensors + llava GGUF)
        {
            "id": 4, "type": "DualCLIPLoaderGGUF",
            "pos": [80, 760], "size": {"0": 460, "1": 100},
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [4, 5], "slot_index": 0}],
            "widgets_values": [
                "clip_l.safetensors",
                "llava-llama-3-8b-v1_1.Q4_0.gguf",
                "hunyuan_video",
            ],
        },
        # 5 — Positive prompt
        {
            "id": 5, "type": "CLIPTextEncode",
            "pos": [640, 300], "size": {"0": 440, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
            "widgets_values": ["A cinematic shot, ultra high resolution, photorealistic"],
        },
        # 6 — Negative prompt
        {
            "id": 6, "type": "CLIPTextEncode",
            "pos": [640, 500], "size": {"0": 440, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [7], "slot_index": 0}],
            "widgets_values": ["text, watermark, low quality, blurry, distorted, artifacts"],
        },
        # 7 — Empty latent at target 1080p
        {
            "id": 7, "type": "EmptyHunyuanLatentVideo",
            "pos": [640, 700], "size": {"0": 320, "1": 120},
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [8], "slot_index": 0}],
            "widgets_values": [TARGET_WIDTH, TARGET_HEIGHT, TARGET_FRAMES, 1],
        },
        # 8 — Sampler (not distilled: more steps, higher cfg)
        {
            "id": 8, "type": "KSampler",
            "pos": [1060, 260], "size": {"0": 320, "1": 280},
            "inputs": [
                {"name": "model",        "type": "MODEL",        "link": 3},
                {"name": "positive",     "type": "CONDITIONING", "link": 6},
                {"name": "negative",     "type": "CONDITIONING", "link": 7},
                {"name": "latent_image", "type": "LATENT",       "link": 8},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9], "slot_index": 0}],
            # seed, seed_mode, steps, cfg, sampler, scheduler, denoise
            "widgets_values": [12345, "randomize", 30, 6.0, "euler", "sgm_uniform", 1.0],
        },
        # 9 — Load VAE
        {
            "id": 9, "type": "VAELoader",
            "pos": [80, 940], "size": {"0": 460, "1": 80},
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [10], "slot_index": 0}],
            "widgets_values": ["hunyuan_video_vae_bf16.safetensors"],
        },
        # 10 — Tiled decode (essential at 1080p — non-tiled would OOM)
        {
            "id": 10, "type": "VAEDecodeTiled",
            "pos": [1460, 260], "size": {"0": 320, "1": 120},
            "inputs": [
                {"name": "samples",   "type": "LATENT", "link": 9},
                {"name": "vae",       "type": "VAE",    "link": 10},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [11], "slot_index": 0}],
            # tile_size, overlap, temporal_size, temporal_overlap
            "widgets_values": [256, 64, 64, 8],
        },
        # 11 — Combine → video
        {
            "id": 11, "type": "VHS_VideoCombine",
            "pos": [1860, 260], "size": {"0": 340, "1": 200},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 11}],
            "outputs": [],
            "widgets_values": [TARGET_FPS, 0, "SEGA_720p_spatial", "video/h264-mp4", False, True],
        },
    ],
    "links": [
        [1,   1, 0,  3, 0, "MODEL"],
        [2,   2, 0,  3, 1, "SEGA_HYVIDEO_CONFIG"],
        [3,   3, 0,  8, 0, "MODEL"],
        [4,   4, 0,  5, 0, "CLIP"],
        [5,   4, 0,  6, 0, "CLIP"],
        [6,   5, 0,  8, 1, "CONDITIONING"],
        [7,   6, 0,  8, 2, "CONDITIONING"],
        [8,   7, 0,  8, 3, "LATENT"],
        [9,   8, 0, 10, 0, "LATENT"],
        [10,  9, 0, 10, 1, "VAE"],
        [11, 10, 0, 11, 0, "IMAGE"],
    ],
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

# ──────────────────────────────────────────────────────────────────────────────
# API workflow
# ──────────────────────────────────────────────────────────────────────────────
api = {
    "1": {
        "inputs": {
            "unet_name": "fast-hunyuan-video-t2v-720p-Q8_0.gguf",
        },
        "class_type": "UnetLoaderGGUF",
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
            "temporal_sega_strength":   0.0,
            "spectral_enabled":         True,
        },
        "class_type": "HunyuanVideoSEGAConfig",
    },
    "3": {
        "inputs": {"model": ["1", 0], "sega_config": ["2", 0]},
        "class_type": "HunyuanVideoSEGAPatch",
    },
    "4": {
        "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "llava-llama-3-8b-v1_1.Q4_0.gguf",
            "type":       "hunyuan_video",
        },
        "class_type": "DualCLIPLoaderGGUF",
    },
    "5": {
        "inputs": {
            "text": "A cinematic shot, ultra high resolution, photorealistic",
            "clip": ["4", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "6": {
        "inputs": {
            "text": "text, watermark, low quality, blurry, distorted, artifacts",
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
            "scheduler":    "sgm_uniform",
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
        "inputs": {
            "samples":          ["8", 0],
            "vae":              ["9", 0],
            "tile_size":        256,
            "overlap":          64,
            "temporal_size":    64,
            "temporal_overlap": 8,
        },
        "class_type": "VAEDecodeTiled",
    },
    "11": {
        "inputs": {
            "images":          ["10", 0],
            "frame_rate":      TARGET_FPS,
            "loop_count":      0,
            "filename_prefix": "SEGA_720p_spatial",
            "format":          "video/h264-mp4",
            "pingpong":        False,
            "save_output":     True,
        },
        "class_type": "VHS_VideoCombine",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Field mappings
# ──────────────────────────────────────────────────────────────────────────────
field_mapping = [
    {"local_field": "prompt",          "node_id": 5,  "node_input_name": "text",       "node_name": "CLIPTextEncode (positive)", "default_value": "A cinematic shot, ultra high resolution, photorealistic"},
    {"local_field": "negative_prompt", "node_id": 6,  "node_input_name": "text",       "node_name": "CLIPTextEncode (negative)", "default_value": "text, watermark, low quality, blurry, distorted, artifacts"},
    {"local_field": "seed",            "node_id": 8,  "node_input_name": "seed",       "node_name": "KSampler",                  "default_value": 12345},
    {"local_field": "steps",           "node_id": 8,  "node_input_name": "steps",      "node_name": "KSampler",                  "default_value": 30},
    {"local_field": "guidance",        "node_id": 8,  "node_input_name": "cfg",        "node_name": "KSampler",                  "default_value": 6.0},
    {"local_field": "strength",        "node_id": 8,  "node_input_name": "denoise",    "node_name": "KSampler",                  "default_value": 1.0},
    {"local_field": "size",            "node_id": 7,  "node_input_name": "width",      "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_WIDTH,  "help_text": "Training default: 1280. 1920 = 1.5× SEGA. Max safe: ~2048 on 24GB."},
    {"local_field": "size",            "node_id": 7,  "node_input_name": "height",     "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_HEIGHT, "help_text": "Training default: 720. 1080 = 1.5× SEGA. Max safe: ~1152 on 24GB."},
    {"local_field": "length",          "node_id": 7,  "node_input_name": "length",     "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_FRAMES, "help_text": "Keep at 25 (training default). Temporal SEGA is disabled."},
    {"local_field": "fps",             "node_id": 11, "node_input_name": "frame_rate", "node_name": "VHS_VideoCombine",          "default_value": TARGET_FPS},
]

# ──────────────────────────────────────────────────────────────────────────────
# Graydient config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            f"FastHunyuan 720p (GGUF Q8_0) + SEGA spatial extrapolation: "
            f"{TARGET_WIDTH}×{TARGET_HEIGHT} @ {TARGET_FRAMES} frames. "
            f"Spatial RoPE extrapolated 1.5× beyond training (720p→1080p). "
            f"Temporal axis unchanged. Tiled VAE decode for 1080p output."
        ),
        "peak_vram_usage": 0,
        "requirements": {
            "github": [
                "https://github.com/UnlimitedEditing/Comfyui-Sega-HyVideo",
                "https://github.com/city96/ComfyUI-GGUF",
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

out = "GraydientWorkflow-sega-hyvideo-v2-720p-spatial.json"
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written : {out}")
print(f"Model   : FastHunyuan 720p Q8_0 GGUF (original HunyuanVideo architecture)")
print(f"Target  : {TARGET_WIDTH}×{TARGET_HEIGHT} @ {TARGET_FRAMES} frames / {TARGET_FPS} fps")
print(f"Training: {TRAIN_WIDTH}×{TRAIN_HEIGHT}, t_len={TRAIN_T_LEN}")
print(f"SEGA    : spatial {TARGET_WIDTH/TRAIN_WIDTH:.2f}×W {TARGET_HEIGHT/TRAIN_HEIGHT:.2f}×H  |  temporal: disabled")
print()
print("IMPORTANT - if node class_type errors appear on Graydient:")
print("  UnetLoaderGGUF     -> check ComfyUI-GGUF is installed and class name is correct")
print("  DualCLIPLoaderGGUF -> may be 'DualCLIPLoader (GGUF)' in UI but same class_type")
