import torch
from typing import Sequence, Optional, Literal
from models.points_sampler import PropPointSampler
from nerfacc.volrend import rendering
from loguru import logger


def render_image_with_propnet(
    # scene
    radiance_field,
    sampler: PropPointSampler,
    rays,
    # rendering options
    num_samples: int,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
    training: bool = True,
):
    """Render the pixels of an image."""
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = rays_o[..., None, :]
        t_dirs = rays_d[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays_o[..., None, :]
        t_dirs = rays_d[..., None, :].repeat_interleave(t_starts.shape[-1], dim=-2)
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    # results = []
    t_starts, t_ends = sampler.estimator.sampling(
        prop_sigma_fns=[
            lambda *args: prop_sigma_fn(*args, p) for p in sampler.proposal_networks
        ],
        prop_samples=sampler.num_samples_per_prop,
        num_samples=num_samples,
        n_rays=n_rays,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        stratified=training,
        requires_grad=proposal_requires_grad,
    )
    rgb, opacity, depth, extras = rendering(
        t_starts,
        t_ends,
        ray_indices=None,
        n_rays=None,
        rgb_sigma_fn=rgb_sigma_fn,
        render_bkgd=render_bkgd,
    )
    # chunk_results = [rgb, opacity, depth]
    # results.append(chunk_results)

    # colors, opacities, depths = collate(
    #     results,
    #     collate_fn_map={
    #         **default_collate_fn_map,
    #         torch.Tensor: lambda x, **_: torch.cat(x, 0),
    #     },
    # )

    sampler.estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )
    return (
        rgb.view((n_rays, -1)),
        opacity.view((n_rays, -1)),
        depth.view((n_rays, -1)),
        extras,
    )
