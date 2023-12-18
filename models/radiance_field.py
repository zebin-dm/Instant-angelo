import torch
import numpy as np
from typing import List, Union, Callable
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


class NGPDensityField(torch.nn.Module):
    """Instance-NGP Density Field used for resampling"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

    def forward(self, positions: torch.Tensor):
        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        density_before_activation = (
            self.mlp_base(positions.view(-1, self.num_dim))
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = (
            self.density_activation(density_before_activation) * selector[..., None]
        )
        return density
