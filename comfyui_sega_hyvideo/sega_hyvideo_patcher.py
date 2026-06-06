"""
SEGA patcher for HunyuanVideo.

Strategy:
  1. Register a forward hook on transformer.img_in to capture the embedded
     hidden states + patch-space T/H/W dimensions after the first img_in call.
  2. Replace transformer.pe_embedder with SEGAEmbedND3D, which:
       a. Scales position coordinates by 1/ntk_factor per axis (equivalent to
          NTK theta scaling — mathematically identical, no recompute needed).
       b. Applies per-dim SEGA mscale as a gain on the 2x2 rotation matrices.
  3. SEGAEmbedND3D resets the shared state in its finally block so each
     denoising step starts clean.

HunyuanVideo axes_dim = [16, 56, 56]  →  T | H | W
pe output shape: (B, 1, n_tokens, 64, 2, 2)
  T slice : [:, :, :, 0:8,   :, :]
  H slice : [:, :, :, 8:36,  :, :]
  W slice : [:, :, :, 36:64, :, :]
"""

import math
import torch
import torch.nn as nn

from .sega_spectral_3d import (
    compute_ntk_factor,
    compute_base_mscale,
    compute_sega_allocation,
    compute_dynamic_spread,
    compute_3d_axis_spectral_profiles,
)


# ---------------------------------------------------------------------------
# Shared state (set by img_in hook, read by SEGAEmbedND3D, reset in finally)
# ---------------------------------------------------------------------------

class SEGAState:
    def __init__(self):
        self.reset()

    def reset(self):
        self._hook_call_count = 0
        self.img_hidden = None
        self.t_len = None
        self.h_len = None
        self.w_len = None
        self.valid  = False


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _get_hyvideo_transformer(model):
    """
    Walk common ComfyUI model-wrapper paths to find the HunyuanVideo transformer.
    Identified by: has both double_blocks and pe_embedder attributes.
    """
    def _is_hyvideo(m):
        return (m is not None
                and hasattr(m, 'double_blocks')
                and hasattr(m, 'pe_embedder'))

    # model → model.model → diffusion_model / model
    for attr1 in ('model',):
        inner = getattr(model, attr1, None)
        if inner is None:
            continue
        if _is_hyvideo(inner):
            return inner
        for attr2 in ('diffusion_model', 'model'):
            dm = getattr(inner, attr2, None)
            if _is_hyvideo(dm):
                return dm
            # One more level
            if dm is not None:
                for attr3 in ('diffusion_model', 'model'):
                    dm2 = getattr(dm, attr3, None)
                    if _is_hyvideo(dm2):
                        return dm2
    return None


# ---------------------------------------------------------------------------
# rope() fallback (matches comfy.ldm.flux.math.rope output format exactly)
# ---------------------------------------------------------------------------

def _rope_fallback(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Fallback rope implementation.
    Matches comfy output: (..., dim//2, 2, 2) float32 rotation matrices.
    """
    assert dim % 2 == 0
    scale = torch.linspace(
        0, (dim - 2) / dim, dim // 2,
        dtype=torch.float64, device=pos.device,
    )
    omega = 1.0 / (theta ** scale)
    out   = torch.einsum("...n,d->...nd", pos.double(), omega)
    # Build [[cos, -sin], [sin, cos]] rotation matrices
    cos_o = torch.cos(out)
    sin_o = torch.sin(out)
    mat = torch.stack([
        torch.stack([ cos_o, -sin_o], dim=-1),   # row 0: [cos, -sin]
        torch.stack([ sin_o,  cos_o], dim=-1),   # row 1: [sin,  cos]
    ], dim=-2)                                    # (..., dim//2, 2, 2)
    return mat.float()


def _get_rope():
    try:
        from comfy.ldm.flux.math import rope
        return rope
    except ImportError:
        return _rope_fallback


# ---------------------------------------------------------------------------
# SEGA-aware EmbedND replacement
# ---------------------------------------------------------------------------

class SEGAEmbedND3D(nn.Module):
    """
    Drop-in replacement for HunyuanVideo's pe_embedder (EmbedND).

    Per-axis NTK scaling: pos_scaled = pos / ntk_factor
    (Mathematically identical to using theta * ntk_factor as the base.)

    SEGA mscale: per-dim gain on the 2×2 rotation matrices,
    derived from the spatial-frequency energy of the latent hidden states.
    """

    def __init__(self, original, config: dict, state: SEGAState, patch_size: list):
        super().__init__()
        self.original   = original
        self.config     = config
        self.state      = state
        self.patch_size = patch_size

        self.theta    = original.theta
        self.axes_dim = original.axes_dim  # e.g. [16, 56, 56]

        # Training reference dimensions (patch-space)
        #   h_len_train = training_height / (VAE_spatial × patch_h) = height / 16
        #   w_len_train = training_width  / 16
        self.t_len_train = config.get('training_t_len', 7)
        self.h_len_train = config.get('training_height', 720)  // 16
        self.w_len_train = config.get('training_width',  1280) // 16

        # SEGA hyperparameters
        self.mscale_alpha  = config.get('mscale_alpha', 0.15)
        self.mscale_beta   = config.get('mscale_beta',  1.5)
        self.mscale_min    = config.get('mscale_min',   1.0)
        self.mscale_formula = config.get('base_mscale_formula', 'power_res')
        self.mscale_coeff  = config.get('base_mscale_coefficient', 0.08)
        self.spread_min    = config.get('spread_min',   0.0)
        self.spread_max    = config.get('spread_max',   1.0)
        self.spread_alpha  = config.get('spread_alpha', 1.5)
        self.temporal_str  = config.get('temporal_sega_strength', 0.5)
        self.spectral_on   = config.get('spectral_enabled', True)

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        try:
            return self._sega_forward(ids)
        except Exception as exc:
            import traceback
            print(f"[SEGA-HYVideo] pe_embedder error — falling back to original: {exc}")
            traceback.print_exc()
            return self.original(ids)
        finally:
            self.state.reset()

    # ------------------------------------------------------------------
    def _sega_forward(self, ids: torch.Tensor) -> torch.Tensor:
        state = self.state

        if not state.valid:
            return self.original(ids)

        t_len, h_len, w_len = state.t_len, state.h_len, state.w_len

        # --- Scale ratios ---
        s_h = max(h_len / max(self.h_len_train, 1), 1.0)
        s_w = max(w_len / max(self.w_len_train, 1), 1.0)
        s_t = max(t_len / max(self.t_len_train, 1), 1.0)
        s_spatial = max(s_h, s_w)

        d_t  = self.axes_dim[0]   # 16 — temporal RoPE dim
        d_sp = self.axes_dim[1]   # 56 — spatial  RoPE dim

        ntk_sp = compute_ntk_factor(s_spatial, d_sp)
        ntk_t  = compute_ntk_factor(s_t, d_t) if self.temporal_str > 0 else 1.0

        # No extrapolation on any axis → pass through unchanged
        if ntk_sp <= 1.0 + 1e-4 and ntk_t <= 1.0 + 1e-4:
            return self.original(ids)

        print(f"[SEGA-HYVideo] step: t={t_len} h={h_len} w={w_len} "
              f"ntk_sp={ntk_sp:.3f} ntk_t={ntk_t:.3f}")

        # --- Per-dim mscale ---
        ms_t, ms_h, ms_w = self._compute_mscales(
            state, s_t, s_h, s_w, ntk_t, ntk_sp, d_t, d_sp
        )

        # --- Build pe axis by axis ---
        _rope = _get_rope()
        n_axes = ids.shape[-1]
        embs   = []

        for i in range(n_axes):
            pos  = ids[..., i]
            d_i  = self.axes_dim[i]

            if i == 0:          # Temporal
                ntk_i = ntk_t
                ms_i  = ms_t
            elif i == 1:        # Height
                ntk_i = ntk_sp
                ms_i  = ms_h
            else:               # Width
                ntk_i = ntk_sp
                ms_i  = ms_w

            # NTK: shrink positions so rope(pos/ntk, θ) ≡ rope(pos, θ·ntk)
            pos_s = pos.float() / ntk_i if ntk_i > 1.0 + 1e-4 else pos

            emb_i = _rope(pos_s, d_i, int(self.theta))   # (*, d_i//2, 2, 2)

            # Apply mscale to rotation matrices
            if ms_i is not None:
                if isinstance(ms_i, torch.Tensor) and ms_i.numel() > 1:
                    ms_t_ = ms_i.to(emb_i.device).to(emb_i.dtype)
                    extra = emb_i.ndim - 3                   # batch dims before (d//2, 2, 2)
                    emb_i = emb_i * ms_t_.view(*([1] * extra), -1, 1, 1)
                else:
                    v = float(ms_i) if not isinstance(ms_i, torch.Tensor) else ms_i.item()
                    if abs(v - 1.0) > 1e-6:
                        emb_i = emb_i * v

            embs.append(emb_i)

        emb = torch.cat(embs, dim=-3)   # (B, n_tokens, 64, 2, 2)
        return emb.unsqueeze(1)         # (B, 1, n_tokens, 64, 2, 2)

    # ------------------------------------------------------------------
    def _compute_mscales(self, state, s_t, s_h, s_w, ntk_t, ntk_sp, d_t, d_sp):
        """
        Try full spectral-energy guided allocation; fall back to global base_mscale.
        Returns (mscale_t, mscale_h, mscale_w) — each is None / scalar / Tensor(d//2).
        """
        ms_t = ms_h = ms_w = None

        if self.spectral_on and state.img_hidden is not None:
            try:
                T, H, W = state.t_len, state.h_len, state.w_len
                n_bt = max(T // 2, 4)
                n_bh = max(H // 2, 4)
                n_bw = max(W // 2, 4)

                pt, ph, pw = compute_3d_axis_spectral_profiles(
                    state.img_hidden, T, H, W, n_bt, n_bh, n_bw,
                )

                if ntk_sp > 1.0 + 1e-4:
                    sp_h  = compute_dynamic_spread(ph, self.spread_min, self.spread_max, self.spread_alpha)
                    sp_w  = compute_dynamic_spread(pw, self.spread_min, self.spread_max, self.spread_alpha)
                    ms_h  = compute_sega_allocation(
                        ph, d_sp // 2,
                        compute_base_mscale(s_h, self.mscale_formula, self.mscale_coeff),
                        sp_h, self.mscale_alpha, self.mscale_beta, self.mscale_min,
                    )
                    ms_w  = compute_sega_allocation(
                        pw, d_sp // 2,
                        compute_base_mscale(s_w, self.mscale_formula, self.mscale_coeff),
                        sp_w, self.mscale_alpha, self.mscale_beta, self.mscale_min,
                    )

                if ntk_t > 1.0 + 1e-4 and self.temporal_str > 0:
                    sp_t = compute_dynamic_spread(pt, self.spread_min, self.spread_max, self.spread_alpha)
                    ms_t = compute_sega_allocation(
                        pt, d_t // 2,
                        compute_base_mscale(s_t, self.mscale_formula, self.mscale_coeff),
                        sp_t * self.temporal_str,
                        self.mscale_alpha, self.mscale_beta, self.mscale_min,
                    )

                return ms_t, ms_h, ms_w

            except Exception as exc:
                print(f"[SEGA-HYVideo] Spectral analysis failed, using base mscale: {exc}")

        # Global fallback
        if ntk_sp > 1.0 + 1e-4:
            ms_h = compute_base_mscale(s_h, self.mscale_formula, self.mscale_coeff)
            ms_w = compute_base_mscale(s_w, self.mscale_formula, self.mscale_coeff)
        if ntk_t > 1.0 + 1e-4 and self.temporal_str > 0:
            ms_t = compute_base_mscale(s_t, self.mscale_formula, self.mscale_coeff) * self.temporal_str

        return ms_t, ms_h, ms_w


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------

class SEGAHunyuanVideoPatcher:
    """
    Patches a HunyuanVideo model with SEGA 3D spectral-energy guided RoPE rescaling.

    Usage:
        patcher = SEGAHunyuanVideoPatcher(model, config)
        patcher.patch()
        # ... run sampling ...
        patcher.unpatch()   # optional — restores original pe_embedder
    """

    def __init__(self, model, config: dict):
        self.model   = model
        self.config  = config
        self.state   = SEGAState()

        self._transformer       = None
        self._original_pe       = None
        self._sega_pe           = None
        self._hook_handle       = None
        self._patched           = False

    # ------------------------------------------------------------------
    def patch(self):
        if self._patched:
            return

        transformer = _get_hyvideo_transformer(self.model)
        if transformer is None:
            raise ValueError(
                "[SEGA-HYVideo] Could not locate HunyuanVideo transformer. "
                "Expected model.model.diffusion_model with double_blocks + pe_embedder."
            )

        patch_size = list(getattr(transformer, 'patch_size', [1, 2, 2]))
        axes_dim   = transformer.pe_embedder.axes_dim

        self._transformer = transformer
        self._original_pe = transformer.pe_embedder
        self.state        = SEGAState()

        # --- Build SEGA pe embedder ---
        self._sega_pe = SEGAEmbedND3D(
            transformer.pe_embedder, self.config, self.state, patch_size
        )

        # --- Hook on img_in: capture dims + hidden states (first call only) ---
        state = self.state
        p     = patch_size

        def _img_in_hook(module, inp, output):
            state._hook_call_count += 1
            if state._hook_call_count != 1:
                return          # ignore ref_latent's second call

            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if x.ndim == 5:    # (B, C, T_lat, H_lat, W_lat)
                state.t_len = (x.shape[2] + (p[0] // 2)) // p[0]
                state.h_len = (x.shape[3] + (p[1] // 2)) // p[1]
                state.w_len = (x.shape[4] + (p[2] // 2)) // p[2]
                state.img_hidden = output.detach()
                state.valid  = True
            elif x.ndim == 4:  # 2-D image fallback (img_ids_2d path)
                state.h_len  = (x.shape[2] + (p[1] // 2)) // p[1]
                state.w_len  = (x.shape[3] + (p[2] // 2)) // p[2]
                state.t_len  = 1
                state.img_hidden = output.detach()
                state.valid  = True

        self._hook_handle = transformer.img_in.register_forward_hook(_img_in_hook)

        # --- Swap pe_embedder ---
        transformer.pe_embedder = self._sega_pe
        self._patched = True

        h = self.config.get('training_height', 720)
        w = self.config.get('training_width',  1280)
        t = self.config.get('training_t_len',  7)
        print(f"[SEGA-HYVideo] Patched  axes_dim={axes_dim}  patch_size={patch_size}")
        print(f"[SEGA-HYVideo] Training ref  {h}×{w}  t_len={t}  "
              f"→ patch coords h={h//16} w={w//16}")

    # ------------------------------------------------------------------
    def unpatch(self):
        if not self._patched:
            return
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._transformer is not None and self._original_pe is not None:
            self._transformer.pe_embedder = self._original_pe
        self._patched = False
        print("[SEGA-HYVideo] Unpatched — original pe_embedder restored.")
