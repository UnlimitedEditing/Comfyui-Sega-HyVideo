"""
ComfyUI nodes for SEGA HunyuanVideo resolution extrapolation.
"""

from .sega_hyvideo_patcher import SEGAHunyuanVideoPatcher


class HunyuanVideoSEGAConfig:
    """
    Configure SEGA spectral-energy guided attention for HunyuanVideo.
    Connect the output to HunyuanVideoSEGAPatch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Training reference ──────────────────────────────────
                "training_height": ("INT", {
                    "default": 720, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Pixel height the model was trained at (720 for HunyuanVideo 720p)",
                }),
                "training_width": ("INT", {
                    "default": 1280, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Pixel width the model was trained at (1280 for HunyuanVideo 720p)",
                }),
                "training_t_len": ("INT", {
                    "default": 7, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Latent temporal length at training time. "
                               "25 input frames ≈ 7 with HunyuanVideo's 4× temporal VAE.",
                }),
                # ── SEGA spectral params ────────────────────────────────
                "mscale_alpha": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sharpness-bias strength — how aggressively high-energy "
                               "RoPE dims are penalised.",
                }),
                "mscale_beta": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Energy standardisation steepness (tanh slope).",
                }),
                "mscale_min": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Floor for per-dim mscale values.",
                }),
                "base_mscale_formula": (["power_res", "log_res"], {
                    "default": "power_res",
                }),
                "base_mscale_coefficient": ("FLOAT", {
                    "default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Exponent/coefficient for the base mscale formula.",
                }),
                "spread_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "spread_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "spread_alpha": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Spread concentration — how sharply spectral "
                               "concentration maps to spread.",
                }),
                # ── Temporal axis control ────────────────────────────────
                "temporal_sega_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much SEGA to apply to the temporal RoPE axis. "
                               "0 = no temporal scaling, 1 = full. "
                               "Set to 0 when frame count matches training.",
                }),
                "spectral_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use spectral energy guidance for per-dim mscale. "
                               "False = NTK scaling only (faster, less precise).",
                }),
            }
        }

    RETURN_TYPES  = ("SEGA_HYVIDEO_CONFIG",)
    RETURN_NAMES  = ("sega_config",)
    FUNCTION      = "make_config"
    CATEGORY      = "SEGA/HunyuanVideo"

    def make_config(
        self,
        training_height, training_width, training_t_len,
        mscale_alpha, mscale_beta, mscale_min,
        base_mscale_formula, base_mscale_coefficient,
        spread_min, spread_max, spread_alpha,
        temporal_sega_strength, spectral_enabled,
    ):
        config = {
            "training_height":           training_height,
            "training_width":            training_width,
            "training_t_len":            training_t_len,
            "mscale_alpha":              mscale_alpha,
            "mscale_beta":               mscale_beta,
            "mscale_min":                mscale_min,
            "base_mscale_formula":       base_mscale_formula,
            "base_mscale_coefficient":   base_mscale_coefficient,
            "spread_min":                spread_min,
            "spread_max":                spread_max,
            "spread_alpha":              spread_alpha,
            "temporal_sega_strength":    temporal_sega_strength,
            "spectral_enabled":          spectral_enabled,
        }
        return (config,)


class HunyuanVideoSEGAPatch:
    """
    Apply SEGA spectral-energy guided RoPE rescaling to a HunyuanVideo model.

    Place AFTER loading your model (UNETLoader / CheckpointLoaderSimple)
    and BEFORE the sampler node.

    At training resolution the patch is a no-op — NTK factor = 1.0 so the
    original pe_embedder output is returned unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":       ("MODEL",),
                "sega_config": ("SEGA_HYVIDEO_CONFIG",),
            }
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "apply_sega"
    CATEGORY      = "SEGA/HunyuanVideo"

    def apply_sega(self, model, sega_config):
        patcher = SEGAHunyuanVideoPatcher(model, sega_config)
        patcher.patch()

        # Hang patcher on the model so it can be unpatched later if needed
        model._sega_hyvideo_patcher = patcher

        h = sega_config["training_height"]
        w = sega_config["training_width"]
        t = sega_config["training_t_len"]
        spectral = "spectral+NTK" if sega_config["spectral_enabled"] else "NTK only"
        print(f"[SEGA-HYVideo] Ready ({spectral})  training ref = {h}×{w}  t_len={t}")

        return (model,)
