"""
HashMedium: instant-NGP hash-grid medium field for WaterSplatting.

Replaces the direction-conditioned MLP with a spatially-varying field:
    raw = hash_medium(xyz)   # xyz: (N, 3) Gaussian world positions
    raw: (N, 9)
    raw[:, :3]  → B_∞   (sigmoid)   — veiling light colour
    raw[:, 3:6] → β_B   (softplus)  — backscatter coefficient
    raw[:, 6:]  → β_D   (softplus)  — attenuation coefficient

Two backends:
    "tcnn"  — tiny-cuda-nn NetworkWithInputEncoding (fast, recommended)
    "torch" — pure-PyTorch fallback (slower, for CPU / debug)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Pure-PyTorch fallback (used when tcnn is unavailable)
class _TorchHashMedium(nn.Module):
    """
    Pure-PyTorch positional encoding + small MLP fallback.

    Uses frequency encoding (NeRF-style) instead of a hash grid.
    Suitable for debugging; does NOT match the speed of tcnn.
    """

    def __init__(
        self,
        n_freq_bands: int = 6,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        # Frequency encoding: xyz → [sin/cos at 2^0…2^(L-1)] + raw xyz
        self.n_freq_bands = n_freq_bands
        in_dim = 3 + 3 * 2 * n_freq_bands  # raw xyz + sin/cos bands

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 9))
        self.net = nn.Sequential(*layers)

    def _encode(self, xyz: Tensor) -> Tensor:
        """Frequency-encode xyz coordinates."""
        freqs = 2.0 ** torch.arange(self.n_freq_bands, device=xyz.device, dtype=xyz.dtype)
        # xyz: (N, 3), freqs: (L,) → (N, 3, L)
        angles = xyz.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * torch.pi
        encoded = torch.cat([xyz, angles.sin().flatten(1), angles.cos().flatten(1)], dim=-1)
        return encoded

    def forward(self, xyz_norm: Tensor) -> Tensor:
        """
        Args:
            xyz_norm: (N, 3) positions already normalised to [0, 1].
        Returns:
            (N, 9) raw network outputs (pre-activation).
        """
        return self.net(self._encode(xyz_norm))


# tiny-cuda-nn backend
def _build_tcnn_network(
    n_levels: int,
    n_features_per_level: int,
    log2_hashmap_size: int,
    base_resolution: int,
    per_level_scale: float,
    hidden_dim: int,
    n_hidden_layers: int,
) -> nn.Module:
    """Build a tcnn NetworkWithInputEncoding (hash grid + MLP fused)."""
    try:
        import tinycudann as tcnn
    except ImportError as e:
        raise ImportError(
            "tiny-cuda-nn is required for mlp_type='tcnn'. "
            "Install it from https://github.com/NVlabs/tiny-cuda-nn, "
            "or set mlp_type='torch' in WaterSplattingModelConfig."
        ) from e

    return tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=9,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
            "interpolation": "Linear",
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
        },
    )

# The main module
class HashMedium(nn.Module):
    """
    Spatial water-medium field using an instant-NGP hash grid.

    Maps 3D Gaussian world positions → 9 raw medium parameters:
        out[:, :3]  → B_∞   (veiling light colour,   apply sigmoid)
        out[:, 3:6] → β_B   (backscatter coefficient, apply softplus)
        out[:, 6:]  → β_D   (attenuation coefficient, apply softplus)

    Args:
        aabb_scale:             Half-side of the scene bounding cube (world
                                units).  Positions outside are clamped.
        n_levels:               Hash-grid levels (default 16).
        n_features_per_level:   Features per level (default 2).
        log2_hashmap_size:      Log₂ of hash table size per level (default 19).
        base_resolution:        Coarsest grid resolution (default 16).
        per_level_scale:        Growth factor between levels (default 1.5).
        hidden_dim:             MLP hidden width (default 64).
        n_hidden_layers:        MLP hidden depth (default 2).
        implementation:         "tcnn" (default) or "torch".
    """

    def __init__(
        self,
        aabb_scale: float = 4.0,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        implementation: str = "tcnn",
    ):
        super().__init__()

        self.aabb_scale = aabb_scale
        self.implementation = implementation

        if implementation == "tcnn":
            self.network = _build_tcnn_network(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                per_level_scale=per_level_scale,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
            )
        elif implementation == "torch":
            self.network = _TorchHashMedium(
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
            )
        else:
            raise ValueError(
                f"Unknown implementation '{implementation}'. "
                "Choose 'tcnn' or 'torch'."
            )


    def _normalize(self, xyz: Tensor) -> Tensor:
        """
        Map world-space xyz to [0, 1]³ using the scene AABB.

        The cube extends from -aabb_scale to +aabb_scale on all axes.
        Points outside are clamped to the boundary.
        """
        xyz_norm = (xyz + self.aabb_scale) / (2.0 * self.aabb_scale)
        return xyz_norm.clamp(0.0, 1.0)


    def forward(self, xyz: Tensor) -> Tensor:
        """
        Query the medium field at Gaussian world positions.

        Args:
            xyz: (N, 3) world-space Gaussian positions (float32).

        Returns:
            raw: (N, 9) raw network outputs — activations are applied
                 in WaterSplattingModel.get_outputs(), not here, so that
                 medium_density_bias can still be added before softplus.
        """
        xyz_norm = self._normalize(xyz)

        if self.implementation == "tcnn":
            # tcnn expects float32 and contiguous input
            raw = self.network(xyz_norm.contiguous().float())
            return raw.float()          # tcnn may output half; cast back
        else:
            return self.network(xyz_norm)