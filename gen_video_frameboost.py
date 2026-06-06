"""
Generate GraydientWorkflow-video-frameboost-v1.json

RIFE frame interpolation on any Graydient video.
Designed to be chained AFTER video-upscale-2mpx.

Budget estimate:
  400 frames at 1920×1080 (post-upscale) → 2x RIFE → 799 frames
  ~50ms/pair × 400 pairs = 20s + overhead = ~35s  ✓ trivially fast
  4x RIFE on same: ~65s  ✓ still within budget

Node map:
  1  VHS_LoadVideoUpload     load video at native resolution (no resize)
  2  RIFE VFI                frame interpolation (multiplier via slot1)
  3  VHS_VideoCombine        output at user-specified fps

slot1 = interpolation multiplier (2 or 4)
fps   = OUTPUT fps (set to source_fps × multiplier)
        e.g. 24fps source with 2x RIFE → set fps=48
             24fps source with 4x RIFE → set fps=96

Output frame count:
  2x: N + (N-1) × 1 = 2N-1 frames
  4x: N + (N-1) × 3 = 4N-3 frames
"""

import json

# ──────────────────────────────────────────────────────────────────────────────
# Standard workflow
# ──────────────────────────────────────────────────────────────────────────────
standard = {
    "last_node_id": 3,
    "last_link_id": 2,
    "nodes": [
        # 1 — Load video at native resolution (no resize for frameboost)
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
            "widgets_values": ["", 0, "Disabled", 512, 512, 0, 0, 1],
        },
        # 2 — RIFE frame interpolation
        {
            "id": 2, "type": "RIFE VFI",
            "pos": [640, 80], "size": {"0": 360, "1": 180},
            "inputs": [{"name": "frames", "type": "IMAGE", "link": 1}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2], "slot_index": 0}],
            # ckpt_name, clear_cache_after_n_frames, multiplier,
            # fast_mode, ensemble, scale_factor
            "widgets_values": ["rife47.pth", 10, 2, True, True, 1.0],
        },
        # 3 — Combine interpolated frames → mp4
        {
            "id": 3, "type": "VHS_VideoCombine",
            "pos": [1060, 80], "size": {"0": 340, "1": 200},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 2}],
            "outputs": [],
            "widgets_values": [48, 0, "frameboost", "video/h264-mp4", False, True],
        },
    ],
    "links": [
        [1,  1, 0,  2, 0, "IMAGE"],
        [2,  2, 0,  3, 0, "IMAGE"],
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
            "force_size":          "Disabled",
            "custom_width":        512,
            "custom_height":       512,
            "frame_load_cap":      0,
            "skip_first_frames":   0,
            "select_every_nth":    1,
        },
        "class_type": "VHS_LoadVideoUpload",
    },
    "2": {
        "inputs": {
            "ckpt_name":                  "rife47.pth",
            "frames":                     ["1", 0],
            "clear_cache_after_n_frames": 10,
            "multiplier":                 2,
            "fast_mode":                  True,
            "ensemble":                   True,
            "scale_factor":               1.0,
        },
        "class_type": "RIFE VFI",
    },
    "3": {
        "inputs": {
            "images":          ["2", 0],
            "frame_rate":      48,
            "loop_count":      0,
            "filename_prefix": "frameboost",
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
        "help_text": "Reply to any Graydient video. Works at any resolution.",
    },
    {
        "local_field": "fps",
        "node_id": 3,
        "node_input_name": "frame_rate",
        "node_name": "VHS_VideoCombine",
        "default_value": 48,
        "help_text": (
            "OUTPUT playback fps. Set to source_fps × multiplier. "
            "24fps source + 2x RIFE = 48. 24fps source + 4x RIFE = 96."
        ),
    },
    {
        "local_field": "slot1",
        "node_id": 2,
        "node_input_name": "multiplier",
        "node_name": "RIFE VFI",
        "default_value": 2,
        "help_text": "Frame count multiplier: 2 or 4. Budget: 2x=~35s, 4x=~65s for 400 frames.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Graydient config
# ──────────────────────────────────────────────────────────────────────────────
cfg = {
    "graydient_workflow": {
        "version": 1,
        "description": (
            "RIFE frame interpolation — 2× or 4× frame count. "
            "Works at any resolution. ~35s for 400 frames at 1080p (2×). "
            "Chain AFTER video-upscale-2mpx for best results. "
            "Set fps to source_fps × multiplier (e.g. 24fps + 2x = 48fps)."
        ),
        "peak_vram_usage": 0,
        "requirements": {
            "github": [
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
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

out = "GraydientWorkflow-video-frameboost-v1.json"
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Written : {out}")
print(f"RIFE    : rife47.pth, default 2x multiplier")
print(f"Budget  : 400 frames 2x = ~35s  |  400 frames 4x = ~65s")
print()
print("fps field = OUTPUT fps (source_fps x multiplier)")
print("  24fps source + 2x slot1 -> set fps=48")
print("  24fps source + 4x slot1 -> set fps=96")
