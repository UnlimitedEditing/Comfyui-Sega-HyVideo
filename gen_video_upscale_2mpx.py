"""
Generate GraydientWorkflow-video-upscale-2mpx-v1.json

Upscales any Graydient video to 2Mpx (1920×1080).

Strategy: resize input to exactly 480×270 inside VHS_LoadVideoUpload
(1/4 of target in each dimension), then 4× GAN upscale → exactly 1920×1080.
GAN always processes the same pixel count regardless of source resolution,
so per-frame cost is constant.

Budget estimate:
  400 frames × ~0.35s/frame GAN (RTX 4090) = 140s
  + ~20s overhead (load/encode)
  Total: ~160s  ✓ inside 3-minute limit

Node map:
  1  VHS_LoadVideoUpload     force-resize to 480×270 on load
  2  UpscaleModelLoader      4x-ClearRealityV1.pth
  3  ImageUpscaleWithModel   480×270 → 1920×1080
  4  VHS_VideoCombine        output at source fps

NOTE: No RIFE in this card. Chain video-frameboost AFTER this
      to double frames on already-upscaled 1920×1080 output.
      Running RIFE before this would double the GAN workload.
"""

import json

TARGET_W = 1920
TARGET_H = 1080
GAN_INPUT_W = TARGET_W // 4   # 480
GAN_INPUT_H = TARGET_H // 4   # 270

assert TARGET_W % 4 == 0, "TARGET_W must be divisible by 4 (4× upscale model)"
assert TARGET_H % 4 == 0, "TARGET_H must be divisible by 4 (4× upscale model)"

# ──────────────────────────────────────────────────────────────────────────────
# Standard workflow  (drag-in UI)
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 4,
    "last_link_id": 3,
    "nodes": [
        # 1 — Load video, force-resize to 480×270 immediately
        {
            "id": 1, "type": "VHS_LoadVideoUpload",
            "pos": [80, 80], "size": {"0": 460, "1": 240},
            "inputs": [],
            "outputs": [
                {"name": "IMAGE",       "type": "IMAGE",    "links": [1], "slot_index": 0},
                {"name": "frame_count", "type": "INT",      "links": [],  "slot_index": 1},
                {"name": "audio",       "type": "AUDIO",    "links": [],  "slot_index": 2},
                {"name": "video_info",  "type": "VHS_VIDEOINFO", "links": [], "slot_index": 3},
            ],
            # video, force_rate, force_size, custom_width, custom_height,
            # frame_load_cap, skip_first_frames, select_every_nth
            "widgets_values": [
                "", 0, "Custom", GAN_INPUT_W, GAN_INPUT_H, 0, 0, 1
            ],
        },
        # 2 — Load upscale model
        {
            "id": 2, "type": "UpscaleModelLoader",
            "pos": [80, 380], "size": {"0": 320, "1": 60},
            "inputs": [],
            "outputs": [{"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": [2], "slot_index": 0}],
            "widgets_values": ["4x-ClearRealityV1.pth"],
        },
        # 3 — GAN upscale: 480×270 → 1920×1080
        {
            "id": 3, "type": "ImageUpscaleWithModel",
            "pos": [640, 160], "size": {"0": 320, "1": 80},
            "inputs": [
                {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": 2},
                {"name": "image",         "type": "IMAGE",          "link": 1},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [3], "slot_index": 0}],
            "widgets_values": [],
        },
        # 4 — Combine upscaled frames → mp4
        {
            "id": 4, "type": "VHS_VideoCombine",
            "pos": [1060, 160], "size": {"0": 340, "1": 200},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 3}],
            "outputs": [],
            "widgets_values": [24, 0, "2mpx_upscale", "video/h264-mp4", False, True],
        },
    ],
    "links": [
        [1,  1, 0,  3, 1, "IMAGE"],
        [2,  2, 0,  3, 0, "UPSCALE_MODEL"],
        [3,  3, 0,  4, 0, "IMAGE"],
    ],
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

# ──────────────────────────────────────────────────────────────────────────────
# API workflow
# ──────────────────────────────────────────────────────────────────────────────
api = {
    "1": {
        "inputs": {
            "video":               "",
            "force_rate":          0,
            "force_size":          "Custom",
            "custom_width":        GAN_INPUT_W,
            "custom_height":       GAN_INPUT_H,
            "frame_load_cap":      0,
            "skip_first_frames":   0,
            "select_every_nth":    1,
        },
        "class_type": "VHS_LoadVideoUpload",
    },
    "2": {
        "inputs": {"model_name": "4x-ClearRealityV1.pth"},
        "class_type": "UpscaleModelLoader",
    },
    "3": {
        "inputs": {
            "upscale_model": ["2", 0],
            "image":         ["1", 0],
        },
        "class_type": "ImageUpscaleWithModel",
    },
    "4": {
        "inputs": {
            "images":          ["3", 0],
            "frame_rate":      24,
            "loop_count":      0,
            "filename_prefix": "2mpx_upscale",
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
    {
        "local_field": "init_image_url",
        "node_id": 1,
        "node_input_name": "video",
        "node_name": "VHS_LoadVideoUpload",
        "default_value": "",
        "help_text": "Reply to a Graydient-generated video",
    },
    {
        "local_field": "fps",
        "node_id": 4,
        "node_input_name": "frame_rate",
        "node_name": "VHS_VideoCombine",
        "default_value": 24,
        "help_text": "Output playback fps. Match your source video fps.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Graydient config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            f"GAN upscale to 2Mpx ({TARGET_W}×{TARGET_H}). "
            f"Input is force-resized to {GAN_INPUT_W}×{GAN_INPUT_H} before the "
            f"4× GAN pass, so per-frame cost is constant regardless of source "
            f"resolution. ~160s for 400 frames on RTX 4090. "
            f"Chain with video-frameboost AFTER this for frame doubling."
        ),
        "peak_vram_usage": 0,
        "requirements": {
            "github": [
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
            ],
            "pip": [],
        },
        "field_mapping": field_mapping,
        "workflow":   json.dumps(standard, indent=2),
        "workflow_2": json.dumps(api,      indent=2),
        "supports_img2img":       False,
        "supports_txt2img":       False,
        "install_detected_nodes": True,
        "is_public":              False,
    }
}

out = "GraydientWorkflow-video-upscale-2mpx-v1.json"
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written : {out}")
print(f"Target  : {TARGET_W}x{TARGET_H}  (GAN input: {GAN_INPUT_W}x{GAN_INPUT_H})")
print(f"Model   : 4x-ClearRealityV1.pth")
print(f"Budget  : 400 frames x ~0.35s = 140s + 20s overhead = ~160s")
print()
print("Chain order: video-upscale-2mpx FIRST, then video-frameboost")
print("  Wrong order: frameboost doubles frames -> GAN processes 2x as many")
