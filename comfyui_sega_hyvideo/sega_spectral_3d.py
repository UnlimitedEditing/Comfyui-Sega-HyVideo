"""
SEGA 3D Spectral Analysis
Extension of 2D SEGA spectral helpers to 3D (T, H, W) for video transformers.
Based on: https://github.com/rajabi2001/sega
"""

import math
import torch


# ---------------------------------------------------------------------------
# NTK / base_mscale helpers
# ---------------------------------------------------------------------------

def compute_ntk_factor(scale: float, rope_dim: int) -> float:
    """NTK interpolation factor for a resolution scale ratio and RoPE dimension."""
    if scale <= 1.0 + 1e-6:
        return 1.0
    return (scale ** (2.0 * rope_dim / (rope_dim - 2))) / (1.0 + 0.1 * math.log(scale))


def compute_base_mscale(
    scale: float,
    formula: str = "power_res",
    coefficient: float = 0.08,
) -> float:
    """Global mscale from resolution scale ratio."""
    s = max(float(scale), 1.0)
    if formula == "power_res":
        return s ** coefficient
    if formula == "log_res":
        return 1.0 + coefficient * math.log(s)
    raise ValueError(f"Unknown base_mscale formula: {formula!r}. Use 'power_res' or 'log_res'.")


# ---------------------------------------------------------------------------
# Per-dim SEGA mscale allocation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sega_allocation(
    energy_profile: torch.Tensor,
    n_freq_dims: int,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
) -> torch.Tensor:
    """
    Spectral-Energy Guided per-dim RoPE mscale allocation.

    Maps each RoPE frequency dimension to its corresponding FFT energy bin via
    the log-period relationship (high-frequency RoPE dims → high energy bins),
    then computes a sharpness-biased mscale:
      high-energy dims → lower mscale (preserve sharpness)
      low-energy dims  → higher mscale (smooth extrapolation)

    Returns: (n_freq_dims,) float32 tensor of per-dim mscale values.
    """
    device = energy_profile.device

    if spread <= 0.0 or alpha <= 0.0 or base_mscale <= 1.0 + 1e-8:
        return torch.full((n_freq_dims,), float(base_mscale), device=device, dtype=torch.float32)

    eps = 1e-8
    n_bins = energy_profile.shape[0]

    # In standard RoPE, dim i=0 has the HIGHEST base frequency (omega=1),
    # dim i=n-1 has the LOWEST (omega=1/theta).
    # Map: i=0 → n_bins-1 (high-freq bin), i=n-1 → 0 (low-freq bin).
    dim_indices = torch.arange(n_freq_dims, device=device, dtype=torch.float32)
    bin_pos = (1.0 - dim_indices / (n_freq_dims - 1 + eps)) * (n_bins - 1)

    j_low  = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac   = (bin_pos - j_low.float()).clamp(0.0, 1.0)

    E     = energy_profile.to(device).clamp(min=eps)
    log_E = torch.log(E)
    raw   = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise → tanh → re-centre
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # direction = -1: high-energy dims → lower mscale
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)


# ---------------------------------------------------------------------------
# Dynamic spread
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dynamic_spread(
    energy_profile: torch.Tensor,
    spread_min: float = 0.0,
    spread_max: float = 1.0,
    alpha: float = 1.5,
) -> float:
    """
    Compute spread from spectral flatness of an energy profile.
    Flat spectrum (natural image) → low spread.
    Peaked spectrum (structured content) → high spread.
    """
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean  = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness      = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)


# ---------------------------------------------------------------------------
# 3D axis spectral profiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_3d_axis_spectral_profiles(
    hidden_states: torch.Tensor,
    T: int,
    H: int,
    W: int,
    n_bins_t: int,
    n_bins_h: int,
    n_bins_w: int,
):
    """
    Compute per-axis 1D spectral energy profiles from video hidden states.

    Args:
        hidden_states: (B, T*H*W, C) embedded image tokens (after img_in)
        T, H, W:       patch-space temporal/spatial dimensions
        n_bins_*:      FFT frequency bins per axis

    Returns:
        profile_t: (n_bins_t,)  temporal spectral energy
        profile_h: (n_bins_h,)  height spectral energy
        profile_w: (n_bins_w,)  width spectral energy
    """
    B, S, C = hidden_states.shape
    n_spatial = min(S, T * H * W)

    flat = lambda n: torch.ones(n, dtype=torch.float32, device=hidden_states.device)

    try:
        spatial = hidden_states[:, :n_spatial].reshape(B, T, H, W, C)
    except RuntimeError:
        return flat(n_bins_t), flat(n_bins_h), flat(n_bins_w)

    sm = spatial.float().mean(dim=(0, -1))   # (T, H, W)
    sm = sm - sm.mean()

    def _axis_profile(vol, axis, n_bins, length):
        fft   = torch.fft.fft(vol, dim=axis)
        power = fft.abs().pow(2)
        # Average over all other spatial axes
        other = [d for d in range(vol.ndim) if d != axis]
        power = power.mean(dim=other)
        half  = length // 2 + 1
        power = power[:half]

        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx   = (freq_norm * n_bins).long().clamp(0, n_bins - 1)

        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    profile_t = _axis_profile(sm, axis=0, n_bins=n_bins_t, length=T)
    profile_h = _axis_profile(sm, axis=1, n_bins=n_bins_h, length=H)
    profile_w = _axis_profile(sm, axis=2, n_bins=n_bins_w, length=W)

    return profile_t, profile_h, profile_w
