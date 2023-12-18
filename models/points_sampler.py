import torch
import itertools
from models.radiance_field import NGPDensityField
from nerfacc import PropNetEstimator


class PropPointSampler:
    def __init__(
        self,
        aabb: torch.Tensor,
        unbounded: bool,
        max_steps: int,
        device: torch.device,
        weight_decay: float = 1e-5,
        num_samples_per_prop=[256, 96],
    ):
        self.proposal_networks = [
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=128,
            ).to(device),
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=256,
            ).to(device),
        ]
        optimizer = torch.optim.Adam(
            itertools.chain(
                *[p.parameters() for p in self.proposal_networks],
            ),
            lr=1e-2,
            eps=1e-15,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )

        self.estimator = PropNetEstimator(optimizer, scheduler).to(device)
        self.num_samples_per_prop = num_samples_per_prop

    def train(self):
        for p in self.proposal_networks:
            p.train()
        self.estimator.train()

    def eval(self):
        for p in self.proposal_networks:
            p.eval()
        self.estimator.eval()
