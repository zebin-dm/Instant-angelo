import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import models
from models.base import BaseModel
from models.utils import chunk_batch, ContractionType
from systems.utils import update_module_step
from nerfacc import (
    PropNetEstimator
    OccGridEstimator,
    render_weight_from_density,
    render_weight_from_alpha,
    accumulate_along_rays,
    ray_aabb_intersect,
)

# from nerfacc.intersection import ray_aabb_intersect


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
class NeuSModel(BaseModel):
    def setup(self):
        # object
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
        self.estimator_bg = OccGridEstimator(
            roi_aabb=self.scene_aabb, resolution=128, levels=8
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
        self.render_step_size_bg = 0.001
        # self.render_step_size_bg = self.render_step_size
        print(f"self.render_step_size: {self.render_step_size}")

    def to_device(self, device):
        self.to(device)
        self.estimator.to(device)
        self.estimator_bg.to(device)

    def set_train(self):
        self.train().float()
        self.estimator.train()
        self.estimator_bg.train()

    def set_eval(self):
        self.eval().float()
        self.estimator.eval()
        self.estimator_bg.eval()

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

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[..., None] * self.render_step_size_bg

        if self.training:
            self.estimator.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.001),
            )
            self.estimator_bg.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn_bg,
                occ_thre=self.config.get("grid_prune_occ_thre_bg", 0.01),
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

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts = t_starts[..., None]
            t_ends = t_ends[..., None]
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            density, _ = self.geometry_bg(positions)
            return density

        _, t_maxs, hits = ray_aabb_intersect(
            rays_o, rays_d, self.scene_aabb.view(-1, 6)
        )
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        # near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        t_mins_bg = torch.where(hits, t_maxs, self.near_plane_bg)
        # with torch.no_grad():
        #     ray_indices, t_starts, t_ends = ray_marching(
        #         rays_o,
        #         rays_d,
        #         scene_aabb=None,
        #         grid=self.occupancy_grid_bg if self.config.grid_prune else None,
        #         sigma_fn=sigma_fn,
        #         near_plane=near_plane,
        #         far_plane=self.far_plane_bg,
        #         render_step_size=self.render_step_size_bg,
        #         stratified=self.randomized,
        #         cone_angle=self.cone_angle_bg,
        #         alpha_thre=0.0,
        #     )
        ray_indices, t_starts, t_ends = self.estimator_bg.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            sigma_fn=sigma_fn,
            t_min=t_mins_bg,
            near_plane=self.near_plane_bg,
            far_plane=self.far_plane_bg,
            render_step_size=self.render_step_size_bg,
            stratified=self.training,
            cone_angle=0.004,
            alpha_thre=0.01,
        )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = ((t_starts + t_ends) / 2.0)[..., None]  # Nx1
        positions = t_origins + t_dirs * midpoints
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions)
        rgb = self.texture_bg(feature, t_dirs)

        weights, _, _ = render_weight_from_density(
            t_starts, t_ends, density, ray_indices=ray_indices, n_rays=n_rays
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

        out = {
            "comp_rgb": comp_rgb,
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
                    "weights": weights.view(-1),
                    "points": midpoints.view(-1),
                    "intervals": intervals.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )

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
        # print(
        #     f'n_rays: {n_rays}, fg samples: {out["num_samples"].item()}, bg samples: {out_bg["num_samples"].item()}'
        # )
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
