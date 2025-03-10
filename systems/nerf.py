import torch
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR


@systems.register("nerf-system")
class NeRFSystem(BaseSystem):
    def prepare(self):
        self.criterions = {"psnr": PSNR()}
        self.train_num_samples = (
            self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        )
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch["rays"])

    def preprocess_data(self, batch, stage):
        if "index" in batch:  # validation / testing
            index = batch["index"]
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(self.train_num_rays,),
                    device=self.dataset.all_images.device,
                )
            else:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(1,),
                    device=self.dataset.all_images.device,
                )
        if stage in ["train"]:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0,
                self.dataset.w,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            y = torch.randint(
                0,
                self.dataset.h,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = (
                self.dataset.all_images[index, y, x]
                .view(-1, self.dataset.all_images.shape[-1])
                .to(self.rank)
            )
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = (
                self.dataset.all_images[index]
                .view(-1, self.dataset.all_images.shape[-1])
                .to(self.rank)
            )
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ["train"]:
            if self.config.model.background_color == "white":
                self.model.background_color = torch.ones(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "random":
                self.model.background_color = torch.rand(
                    (3,), dtype=torch.float32, device=self.rank
                )
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones(
                (3,), dtype=torch.float32, device=self.rank
            )

        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[..., None] + self.model.background_color * (
                1 - fg_mask[..., None]
            )

        batch.update({"rays": rays, "rgb": rgb, "fg_mask": fg_mask})

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(
                self.train_num_rays
                * (self.train_num_samples / out["num_samples"].sum().item())
            )
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                self.config.model.max_train_num_rays,
            )

        loss_rgb = F.smooth_l1_loss(
            out["comp_rgb"][out["rays_valid"][..., 0]],
            batch["rgb"][out["rays_valid"][..., 0]],
        )
        self.log("train/loss_rgb", loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(
                out["weights"], out["points"], out["intervals"], out["ray_indices"]
            )
            self.log("train/loss_distortion", loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f"train/loss_{name}", value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.log(f"train_params/{name}", self.C(value))

        self.log("train/num_rays", float(self.train_num_rays), prog_bar=True)

        return {"loss": loss}

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions["psnr"](out["comp_rgb"].to(batch["rgb"]), batch["rgb"])
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "grayscale",
                    "img": out["opacity"].view(H, W),
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
        )
        return {"psnr": psnr, "index": batch["index"]}

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    out_set[step_out["index"].item()] = {"psnr": step_out["psnr"]}
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        out_set[index[0].item()] = {"psnr": step_out["psnr"][oi]}
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))
            self.log("val/psnr", psnr, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions["psnr"](out["comp_rgb"].to(batch["rgb"]), batch["rgb"])
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "grayscale",
                    "img": out["opacity"].view(H, W),
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
        )
        return {"psnr": psnr, "index": batch["index"]}

    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    out_set[step_out["index"].item()] = {"psnr": step_out["psnr"]}
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        out_set[index[0].item()] = {"psnr": step_out["psnr"][oi]}
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))
            self.log("test/psnr", psnr, prog_bar=True, rank_zero_only=True)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
            )

            self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh,
        )
