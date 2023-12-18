import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import models
from models.utils import chunk_batch, ContractionType
from utils.utils import update_module_step
from nerfacc import (
    OccGridEstimator,
    render_weight_from_alpha,
    accumulate_along_rays,
    ray_aabb_intersect,
)
from nerfacc.estimators.prop_net import get_proposal_requires_grad_fn
from models.points_sampler import PropPointSampler
from models.renders import render_image_with_propnet


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter(
            "variance", nn.Parameter(torch.tensor(self.config.init_val))
        )

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        return val

    def forward(self, x):
        return (torch.ones_like(x) * self.inv_s).clip(1e-6, 1e6)


@models.register("neus")
class NeuSModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB
        # backgroud
        self.geometry_bg = models.make(
            self.config.geometry_bg.name, self.config.geometry_bg
        )
        self.texture_bg = models.make(
            self.config.texture_bg.name, self.config.texture_bg
        )
        self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
        self.cone_angle_bg = (
            10 ** (math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg)
            - 1.0
        )
        logger.info(f"===============> self.cone_angle_bg : {self.cone_angle_bg}")
        # self.render_step_size_bg = 0.01

        self.variance = VarianceNetwork(self.config.variance)
        radius = self.config.radius
        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                [
                    -radius,
                    -radius,
                    -radius,
                    radius,
                    radius,
                    radius,
                ],
                dtype=torch.float32,
            ),
        )
        self.estimator = OccGridEstimator(
            roi_aabb=self.scene_aabb, resolution=128, levels=1
        )
        # TODO, remove the paramter
        max_steps = 1600
        self.estimator_bg = PropPointSampler(
            aabb=self.scene_aabb,
            unbounded=True,
            max_steps=max_steps,
            device=self.device,
        )
        self.randomized = self.config.randomized
        # self.render_step_size = (
        #     1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        # )
        # self.render_step_size = 0.005
        # self.render_step_size = (
        #     (self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2
        # ).sum().sqrt().item() / 1000
        self.render_step_size = 0.005
        self.render_step_size_bg = 0.02
        print(f"self.render_step_size: {self.render_step_size}")
        self.current_epoch = None
        self.global_step = None
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

    def to_device(self, device):
        self.to(device)
        self.estimator.to(device)

    def set_train(self):
        self.train().float()
        self.estimator.train()
        self.estimator_bg.train()

    def set_eval(self):
        self.eval().float()
        self.estimator.eval()
        self.estimator_bg.eval()

    def update_status(self, current_epoch, global_step):
        self.current_epoch = current_epoch
        self.global_step = global_step

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)

        update_module_step(self.geometry_bg, epoch, global_step)
        update_module_step(self.texture_bg, epoch, global_step)

        cos_anneal_end = self.config.get("cos_anneal_end", 0)
        self.cos_anneal_ratio = (
            1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)
        )

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)  # N
            sdf = sdf[..., None]  # Nx1
            inv_s = self.variance(sdf)
            estimated_next_sdf = sdf - self.render_step_size * 0.5
            estimated_prev_sdf = sdf + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            return alpha

        if self.training:
            self.estimator.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.001),
            )

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        sdf = sdf[..., None]
        inv_s = self.variance(sdf)
        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        def radiance_field(position, direction):
            density, feature = self.geometry_bg(position)
            rgb = self.texture_bg(feature, direction)
            return rgb, density

        _, t_maxs, hits = ray_aabb_intersect(
            rays_o, rays_d, self.scene_aabb.view(-1, 6)
        )
        proposal_requires_grad = self.proposal_requires_grad_fn(self.global_step)
        rgb, opacity, depth, extras = render_image_with_propnet(
            radiance_field,
            sampler=self.estimator_bg,
            rays=rays,
            num_samples=48,
            near_plane=t_maxs,
            far_plane=torch.ones_like(t_maxs) * self.far_plane_bg,
            sampling_type="lindisp",
            opaque_bkgd=True,
            proposal_requires_grad=proposal_requires_grad,
            training=self.training,
        )
        out = {
            "comp_rgb": rgb,
            "opacity": opacity,
            "depth": depth,
            "rays_valid": opacity > 0,
            "num_samples": torch.as_tensor(
                [len(rays) * 48], dtype=torch.int32, device=rays.device
            ),
        }

        return out

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        def alpha_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * t_positions
            sdf = self.geometry(positions, with_grad=False, with_feature=False)
            inv_std = self.variance(sdf)
            estimated_next_sdf = sdf - self.render_step_size * 0.5
            estimated_prev_sdf = sdf + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

            return alpha

            # with torch.no_grad():
            #     ray_indices, t_starts, t_ends = ray_marching(
            #         rays_o,
            #         rays_d,
            #         scene_aabb=self.scene_aabb,
            #         grid=self.occupancy_grid if self.config.grid_prune else None,
            #         alpha_fn=None,
            #         near_plane=None,
            #         far_plane=None,
            #         render_step_size=self.render_step_size,
            #         stratified=self.randomized,
            #         cone_angle=0.0,
            #         alpha_thre=0.0,
            #     )

        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            alpha_fn=alpha_fn,
            near_plane=0.0,
            far_plane=1e10,
            render_step_size=self.render_step_size,
            stratified=self.training,
            cone_angle=0.0,
            alpha_thre=0.0,
        )

        # def validate_empty_rays(ray_indices, t_start, t_end):
        #     if ray_indices.nelement() == 0:
        #         logger.warning("Empty rays_indices!")
        #         ray_indices = torch.LongTensor([0]).to(ray_indices)
        #         t_start = torch.Tensor([0]).to(ray_indices)
        #         t_end = torch.Tensor([0]).to(ray_indices)
        #     return ray_indices, t_start, t_end

        # ray_indices, t_starts, t_ends = validate_empty_rays(
        #     ray_indices, t_starts, t_ends
        # )
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        t_starts = t_starts[..., None]  # Nx1
        t_ends = t_ends[..., None]  # Nx1
        midpoints = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        sdf, sdf_grad, feature, sdf_laplace = self.geometry(
            positions, with_grad=True, with_feature=True, with_laplace=True
        )

        normal = F.normalize(sdf_grad, p=2, dim=-1)  # N x 3
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)  # N x 1
        rgb = self.texture(feature, t_dirs, normal)
        weights, _ = render_weight_from_alpha(
            alpha[..., 0], ray_indices=ray_indices, n_rays=n_rays
        )
        opacity = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth = accumulate_along_rays(
            weights, values=midpoints, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb = accumulate_along_rays(
            weights, values=rgb, ray_indices=ray_indices, n_rays=n_rays
        )

        comp_normal = accumulate_along_rays(
            weights, values=normal, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            "comp_rgb": comp_rgb,
            "comp_normal": comp_normal,
            "opacity": opacity,
            "depth": depth,
            "rays_valid": opacity > 0,
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.training:
            out.update(
                {
                    "sdf_samples": sdf,
                    "sdf_grad_samples": sdf_grad,
                    "weights": weights,
                    "points": midpoints.view(-1),
                    "intervals": dists.view(-1),
                    "ray_indices": ray_indices,
                }
            )
            out.update({"sdf_laplace_samples": sdf_laplace})

        out_bg = self.forward_bg_(rays)

        out_full = {
            "comp_rgb": out["comp_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"]),
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
        }

        return {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {**out, "inv_s": self.variance.inv_s}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config, device):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, features = chunk_batch(
                self.geometry,
                export_config.chunk_size,
                False,
                mesh["v_pos"].to(device),
                with_grad=True,
                with_feature=True,
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            base_color = torch.sigmoid(features[..., 1:4])
            mesh["v_rgb"] = base_color.cpu()
            mesh["v_norm"] = normal.cpu()
        return mesh
