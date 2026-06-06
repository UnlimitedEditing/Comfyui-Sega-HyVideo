"""
Generate GraydientWorkflow-sega-hyvideo-v1-480p-temporal.json

Workflow 1: HunyuanVideo 1.5 (480p) + SEGA temporal extension
  - Spatial:  640×448 → 1280×720  (~1.6× / 2× per axis — spatial as a bonus)
  - Temporal: 121 frames → 241 frames  (2× — the main goal, extends T-axis RoPE)

SEGA temporal NTK:  s_t = 61/31 ≈ 1.97,  d_t = 16  →  ntk_t ≈ 3.75
SEGA spatial NTK:   s_w = 80/40 = 2.0,   d_sp = 56  →  ntk_sp ≈ 5.1

NOTE: HunyuanVideo 1.5 uses a different architecture from v1. If the SEGA patch
      fails to find double_blocks + pe_embedder it will raise a clear ValueError —
      fall back to Workflow 2 in that case.

Node map:
    1  UNETLoader                    (HunyuanVideo 1.5 diffusion model)
    2  LoraLoaderModelOnly           (LightX2V 4-step distillation LoRA — required)
    3  HunyuanVideoSEGAConfig
    4  HunyuanVideoSEGAPatch
    5  DualCLIPLoader                (Qwen2.5-VL 7B + ByT5 Small)
    6  CLIPTextEncode  (positive)
    7  CLIPTextEncode  (negative)
    8  EmptyHunyuanLatentVideo       (may need renaming for v1.5 — check class_type)
    9  KSampler                      (4 steps, lcm sampler, guidance ~1)
    10 VAELoader
    11 VAEDecodeTiled
    12 VHS_VideoCombine
"""

import json

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_HEIGHT = 720
TARGET_WIDTH  = 1280
TARGET_FRAMES = 241    # 2× extension of training 121 frames
TARGET_FPS    = 24

# ── Training reference for SEGA ────────────────────────────────────────────
# HunyuanVideo 1.5 480p defaults: 640×448, 121 frames
# Latent t_len = (121-1)//4 + 1 = 31  (4× causal VAE, first frame + (rest/4))
TRAIN_HEIGHT  = 448
TRAIN_WIDTH   = 640
TRAIN_T_LEN   = 31

# HunyuanVideo constraint: height & width must be multiples of 16
# (8× VAE spatial compression × 2 patch stride — both must divide evenly)
assert TARGET_HEIGHT % 16 == 0, f"TARGET_HEIGHT {TARGET_HEIGHT} must be divisible by 16"
assert TARGET_WIDTH  % 16 == 0, f"TARGET_WIDTH {TARGET_WIDTH} must be divisible by 16"
assert TRAIN_HEIGHT  % 16 == 0, f"TRAIN_HEIGHT {TRAIN_HEIGHT} must be divisible by 16"
assert TRAIN_WIDTH   % 16 == 0, f"TRAIN_WIDTH {TRAIN_WIDTH} must be divisible by 16"

# ── Derived ────────────────────────────────────────────────────────────────
TARGET_T_LEN  = (TARGET_FRAMES - 1) // 4 + 1   # = 61

# ──────────────────────────────────────────────────────────────────────────────
# Standard workflow
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 12,
    "last_link_id": 15,
    "nodes": [
        # 1 — Load diffusion model weights
        {
            "id": 1, "type": "UNETLoader",
            "pos": [80, 80], "size": {"0": 460, "1": 80},
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "widgets_values": [
                "hunyuanvideo1.5_480p_t2v_fp16.safetensors",
                "default",
            ],
        },
        # 2 — LightX2V LoRA (enables 4-step distillation — required)
        {
            "id": 2, "type": "LoraLoaderModelOnly",
            "pos": [80, 210], "size": {"0": 460, "1": 100},
            "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [2], "slot_index": 0}],
            "widgets_values": [
                "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_bf16.safetensors",
                1.0,   # lora strength (slot1 in Graydient)
            ],
        },
        # 3 — SEGA config (temporal extension focus)
        {
            "id": 3, "type": "HunyuanVideoSEGAConfig",
            "pos": [80, 360], "size": {"0": 460, "1": 480},
            "inputs": [],
            "outputs": [{"name": "SEGA_HYVIDEO_CONFIG", "type": "SEGA_HYVIDEO_CONFIG", "links": [3], "slot_index": 0}],
            "widgets_values": [
                TRAIN_HEIGHT, TRAIN_WIDTH, TRAIN_T_LEN,
                0.15, 1.5, 1.0,       # mscale_alpha / beta / min
                "power_res", 0.08,    # formula, coefficient
                0.0, 1.0, 1.5,        # spread_min / max / alpha
                1.0, True,             # temporal_sega_strength=1.0 (full), spectral=True
            ],
        },
        # 4 — Apply SEGA patch
        {
            "id": 4, "type": "HunyuanVideoSEGAPatch",
            "pos": [640, 200], "size": {"0": 300, "1": 80},
            "inputs": [
                {"name": "model",       "type": "MODEL",               "link": 2},
                {"name": "sega_config", "type": "SEGA_HYVIDEO_CONFIG", "link": 3},
            ],
            "outputs": [{"name": "model", "type": "MODEL", "links": [4], "slot_index": 0}],
            "widgets_values": [],
        },
        # 5 — Load CLIP (Qwen2.5-VL + ByT5 — HunyuanVideo 1.5 text encoders)
        {
            "id": 5, "type": "DualCLIPLoader",
            "pos": [80, 900], "size": {"0": 460, "1": 100},
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [5, 6], "slot_index": 0}],
            "widgets_values": [
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "byt5_small_glyphxl_fp16.safetensors",
                "hunyuan_video",
            ],
        },
        # 6 — Positive prompt
        {
            "id": 6, "type": "CLIPTextEncode",
            "pos": [640, 420], "size": {"0": 440, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [7], "slot_index": 0}],
            "widgets_values": ["A cinematic shot, ultra high resolution"],
        },
        # 7 — Negative prompt
        {
            "id": 7, "type": "CLIPTextEncode",
            "pos": [640, 620], "size": {"0": 440, "1": 160},
            "inputs": [{"name": "clip", "type": "CLIP", "link": 6}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [8], "slot_index": 0}],
            "widgets_values": ["text, watermark, low quality, blurry, distorted"],
        },
        # 8 — Empty latent (NOTE: may need EmptyHunyuanVideoLatent15 for v1.5)
        {
            "id": 8, "type": "EmptyHunyuanLatentVideo",
            "pos": [640, 820], "size": {"0": 320, "1": 120},
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9], "slot_index": 0}],
            "widgets_values": [TARGET_WIDTH, TARGET_HEIGHT, TARGET_FRAMES, 1],
        },
        # 9 — Sampler (4-step distilled: lcm sampler, simple scheduler, low cfg)
        {
            "id": 9, "type": "KSampler",
            "pos": [1060, 320], "size": {"0": 320, "1": 280},
            "inputs": [
                {"name": "model",        "type": "MODEL",        "link": 4},
                {"name": "positive",     "type": "CONDITIONING", "link": 7},
                {"name": "negative",     "type": "CONDITIONING", "link": 8},
                {"name": "latent_image", "type": "LATENT",       "link": 9},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10], "slot_index": 0}],
            # seed, seed_mode, steps, cfg, sampler, scheduler, denoise
            "widgets_values": [12345, "randomize", 4, 1.0, "lcm", "simple", 1.0],
        },
        # 10 — Load VAE (HunyuanVideo 1.5 VAE)
        {
            "id": 10, "type": "VAELoader",
            "pos": [80, 1060], "size": {"0": 460, "1": 80},
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [11], "slot_index": 0}],
            "widgets_values": ["hunyuanvideo15_vae_fp16.safetensors"],
        },
        # 11 — Tiled decode (essential at 720p+ resolution)
        {
            "id": 11, "type": "VAEDecodeTiled",
            "pos": [1460, 320], "size": {"0": 320, "1": 120},
            "inputs": [
                {"name": "samples",   "type": "LATENT", "link": 10},
                {"name": "vae",       "type": "VAE",    "link": 11},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [12], "slot_index": 0}],
            # tile_size, overlap, temporal_size, temporal_overlap
            "widgets_values": [256, 64, 64, 8],
        },
        # 12 — Combine frames → video
        {
            "id": 12, "type": "VHS_VideoCombine",
            "pos": [1860, 320], "size": {"0": 340, "1": 200},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 12}],
            "outputs": [],
            "widgets_values": [TARGET_FPS, 0, "SEGA_480p_temporal", "video/h264-mp4", False, True],
        },
    ],
    "links": [
        [1,   1, 0,  2, 0, "MODEL"],
        [2,   2, 0,  4, 0, "MODEL"],
        [3,   3, 0,  4, 1, "SEGA_HYVIDEO_CONFIG"],
        [4,   4, 0,  9, 0, "MODEL"],
        [5,   5, 0,  6, 0, "CLIP"],
        [6,   5, 0,  7, 0, "CLIP"],
        [7,   6, 0,  9, 1, "CONDITIONING"],
        [8,   7, 0,  9, 2, "CONDITIONING"],
        [9,   8, 0,  9, 3, "LATENT"],
        [10,  9, 0, 11, 0, "LATENT"],
        [11, 10, 0, 11, 1, "VAE"],
        [12, 11, 0, 12, 0, "IMAGE"],
    ],
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

# ──────────────────────────────────────────────────────────────────────────────
# API workflow
# ──────────────────────────────────────────────────────────────────────────────
api = {
    "1": {
        "inputs": {
            "unet_name":    "hunyuanvideo1.5_480p_t2v_fp16.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
    },
    "2": {
        "inputs": {
            "model":      ["1", 0],
            "lora_name":  "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_bf16.safetensors",
            "strength_model": 1.0,
        },
        "class_type": "LoraLoaderModelOnly",
    },
    "3": {
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
            "temporal_sega_strength":   1.0,
            "spectral_enabled":         True,
        },
        "class_type": "HunyuanVideoSEGAConfig",
    },
    "4": {
        "inputs": {"model": ["2", 0], "sega_config": ["3", 0]},
        "class_type": "HunyuanVideoSEGAPatch",
    },
    "5": {
        "inputs": {
            "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "clip_name2": "byt5_small_glyphxl_fp16.safetensors",
            "type":       "hunyuan_video",
        },
        "class_type": "DualCLIPLoader",
    },
    "6": {
        "inputs": {
            "text": "A cinematic shot, ultra high resolution",
            "clip": ["5", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "7": {
        "inputs": {
            "text": "text, watermark, low quality, blurry, distorted",
            "clip": ["5", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "8": {
        "inputs": {
            "width":      TARGET_WIDTH,
            "height":     TARGET_HEIGHT,
            "length":     TARGET_FRAMES,
            "batch_size": 1,
        },
        "class_type": "EmptyHunyuanLatentVideo",
    },
    "9": {
        "inputs": {
            "seed":         12345,
            "steps":        4,
            "cfg":          1.0,
            "sampler_name": "lcm",
            "scheduler":    "simple",
            "denoise":      1.0,
            "model":        ["4", 0],
            "positive":     ["6", 0],
            "negative":     ["7", 0],
            "latent_image": ["8", 0],
        },
        "class_type": "KSampler",
    },
    "10": {
        "inputs": {"vae_name": "hunyuanvideo15_vae_fp16.safetensors"},
        "class_type": "VAELoader",
    },
    "11": {
        "inputs": {
            "samples":          ["9", 0],
            "vae":              ["10", 0],
            "tile_size":        256,
            "overlap":          64,
            "temporal_size":    64,
            "temporal_overlap": 8,
        },
        "class_type": "VAEDecodeTiled",
    },
    "12": {
        "inputs": {
            "images":          ["11", 0],
            "frame_rate":      TARGET_FPS,
            "loop_count":      0,
            "filename_prefix": "SEGA_480p_temporal",
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
    {"local_field": "prompt",          "node_id": 6,  "node_input_name": "text",           "node_name": "CLIPTextEncode (positive)", "default_value": "A cinematic shot, ultra high resolution"},
    {"local_field": "negative_prompt", "node_id": 7,  "node_input_name": "text",           "node_name": "CLIPTextEncode (negative)", "default_value": "text, watermark, low quality, blurry, distorted"},
    {"local_field": "seed",            "node_id": 9,  "node_input_name": "seed",           "node_name": "KSampler",                  "default_value": 12345},
    {"local_field": "steps",           "node_id": 9,  "node_input_name": "steps",          "node_name": "KSampler",                  "default_value": 4,    "help_text": "4 for distilled LightX2V, up to 20 without LoRA"},
    {"local_field": "guidance",        "node_id": 9,  "node_input_name": "cfg",            "node_name": "KSampler",                  "default_value": 1.0,  "help_text": "Keep near 1.0 for distilled model"},
    {"local_field": "strength",        "node_id": 9,  "node_input_name": "denoise",        "node_name": "KSampler",                  "default_value": 1.0},
    {"local_field": "size",            "node_id": 8,  "node_input_name": "width",          "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_WIDTH},
    {"local_field": "size",            "node_id": 8,  "node_input_name": "height",         "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_HEIGHT},
    {"local_field": "length",          "node_id": 8,  "node_input_name": "length",         "node_name": "EmptyHunyuanLatentVideo",   "default_value": TARGET_FRAMES, "help_text": "Training default: 121. Push to 241 for 2× temporal SEGA."},
    {"local_field": "fps",             "node_id": 12, "node_input_name": "frame_rate",     "node_name": "VHS_VideoCombine",          "default_value": TARGET_FPS},
    {"local_field": "slot1",           "node_id": 2,  "node_input_name": "strength_model", "node_name": "LoraLoaderModelOnly",       "default_value": 1.0,  "help_text": "LightX2V LoRA strength (keep at 1.0 for 4-step)"},
]

# ──────────────────────────────────────────────────────────────────────────────
# Graydient config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            f"HunyuanVideo 1.5 480p + SEGA temporal extension: "
            f"{TARGET_WIDTH}×{TARGET_HEIGHT}, {TARGET_FRAMES} frames ({TARGET_FPS} fps). "
            f"Temporal RoPE extrapolated 2× beyond training (121→{TARGET_FRAMES} frames)."
        ),
        "peak_vram_usage": 0,
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

out = "GraydientWorkflow-sega-hyvideo-v1-480p-temporal.json"
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written : {out}")
print(f"Model   : HunyuanVideo 1.5 480p (4-step distilled + LightX2V LoRA)")
print(f"Target  : {TARGET_WIDTH}×{TARGET_HEIGHT} @ {TARGET_FRAMES} frames / {TARGET_FPS} fps")
print(f"Training: {TRAIN_WIDTH}×{TRAIN_HEIGHT}, t_len={TRAIN_T_LEN} ({(TRAIN_T_LEN-1)*4+1} input frames)")
print(f"SEGA    : spatial {TARGET_WIDTH/TRAIN_WIDTH:.2f}×W {TARGET_HEIGHT/TRAIN_HEIGHT:.2f}×H  |  temporal {TARGET_T_LEN}/{TRAIN_T_LEN} = {TARGET_T_LEN/TRAIN_T_LEN:.2f}×")
