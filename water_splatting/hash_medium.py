"""HashMedium: instant-NGP hash-grid medium field for WaterSplatting.

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

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


def soft_clamp(x: Tensor, limit: float) -> Tensor:
    """Saturate logits near ±limit while preserving non-zero gradients."""
    return limit * torch.tanh(x / limit)


class _TorchHashMedium(nn.Module):
    """Pure-PyTorch frequency encoding + small MLP fallback.

    Uses NeRF-style frequency encoding instead of a hash grid.
    Suitable for debugging; does NOT match the speed of tcnn.
    """

    def __init__(self, n_freq_bands: int = 6, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        self.n_freq_bands = n_freq_bands
        in_dim = 3 + 6 * n_freq_bands
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 9))
        self.net = nn.Sequential(*layers)

    def _encode(self, xyz: Tensor) -> Tensor:
        """Frequency-encode xyz coordinates."""
        freqs = 2.0 ** torch.arange(self.n_freq_bands, device=xyz.device, dtype=xyz.dtype)
        angles = xyz.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * torch.pi
        return torch.cat([xyz, angles.sin().flatten(1), angles.cos().flatten(1)], dim=-1)

    def forward(self, xyz_norm: Tensor) -> Tensor:
        """Args:
            xyz_norm: (N, 3) positions already normalised to [0, 1].
        Returns:
            (N, 9) raw network outputs (pre-activation).
        """
        return self.net(self._encode(xyz_norm))


def _build_tcnn_network(
    n_levels: int,
    n_features_per_level: int,
    log2_hashmap_size: int,
    base_resolution: int,
    per_level_scale: float,
    hidden_dim: int,
    n_hidden_layers: int,
):
    """Build a tcnn NetworkWithInputEncoding (hash grid + MLP fused)."""
    try:
        import tinycudann as tcnn
    except ImportError as e:
        raise ImportError(
            "tiny-cuda-nn is required for implementation='tcnn'. "
            "Install from https://github.com/NVlabs/tiny-cuda-nn, or set hash_implementation='torch'."
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


class HashMedium(nn.Module):
    """Spatial water-medium field using an instant-NGP hash grid.

    Maps 3D Gaussian world positions → 9 raw medium parameters:
        out[:, :3]  → B_∞   (veiling light colour, apply sigmoid)
        out[:, 3:6] → β_B   (backscatter coefficient, apply softplus)
        out[:, 6:]  → β_D   (attenuation coefficient, apply softplus)

    Args:
        aabb_scale: Half-side of the scene bounding cube (world units).
            Positions outside are clamped.
        n_levels: Hash-grid levels (default 16).
        n_features_per_level: Features per level (default 2).
        log2_hashmap_size: Log₂ of hash table size per level (default 19).
        base_resolution: Coarsest grid resolution (default 16).
        per_level_scale: Growth factor between levels (default 1.5).
        hidden_dim: MLP hidden width (default 64).
        n_hidden_layers: MLP hidden depth (default 2).
        implementation: ``"tcnn"`` (default) or ``"torch"``.
        logit_clamp: Soft tanh clamp limit on raw logits (default 10).
            Set to ``None`` or ``<= 0`` to disable.
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
        logit_clamp: Optional[float] = 10.0,
    ):
        super().__init__()
        self.aabb_scale = aabb_scale
        self.implementation = implementation
        self.logit_clamp = logit_clamp

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
                f"Unknown implementation '{implementation}'. Choose 'tcnn' or 'torch'."
            )

    def _normalize(self, xyz: Tensor) -> Tensor:
        """Map world-space xyz to [0, 1]³ using the scene AABB.

        The cube extends from -aabb_scale to +aabb_scale on all axes.
        Points outside are clamped to the boundary.
        """
        xyz_norm = (xyz + self.aabb_scale) / (2.0 * self.aabb_scale)
        return xyz_norm.clamp(0.0, 1.0)

    def forward(self, xyz: Tensor) -> Tensor:
        """Query the medium field at Gaussian world positions.

        Args:
            xyz: (N, 3) world-space Gaussian positions (float32).

        Returns:
            raw: (N, 9) raw network outputs — activations are applied
                in the model (after optional medium_density_bias), not here.
        """
        xyz_norm = self._normalize(xyz)
        if self.implementation == "tcnn":
            raw = self.network(xyz_norm.contiguous().float()).float()
        else:
            raw = self.network(xyz_norm)

        if self.logit_clamp is not None and self.logit_clamp > 0:
            raw = soft_clamp(raw, self.logit_clamp)
        return raw
