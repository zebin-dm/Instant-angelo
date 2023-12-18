import torch
import datasets
import argparse
from pydlutils.torch import seed
from loguru import logger
from datetime import datetime
from utils.misc import load_config
from torch.utils.tensorboard import SummaryWriter
from systems.neus import NeuSSystem


class Trainer:
    def __init__(self, config):
        self.cfg = config.trainer
        self.cfg_global = config
        self.device = torch.device("cuda")
        self.writer = SummaryWriter(f"{config.exp_dir}/{config.trial_name}")
        self.global_step = 0

    def test(self, system, datamodule):
        logger.info("Testing .....")
        dataloader = datamodule.test_dataloader()
        system.model.set_eval()
        for bidx, batch in enumerate(dataloader):
            system.on_test_batch_start(batch, dataloader.dataset)
            system.test_step(batch, bidx)
        system.model.train()

    def export(self, system):
        logger.info("Exporting mesh ...")
        torch.cuda.empty_cache()
        system.model.set_eval()
        system.export()

    def train(self):
        max_epoch = self.cfg.get("max_epoch", 1)
        cfg = self.cfg
        cfg_g = self.cfg_global
        system = NeuSSystem(cfg_g, self.device)
        system.setup(self.writer)
        datamodule = datasets.make(cfg_g.dataset.name, cfg_g.dataset)
        datamodule.setup(stage=None, device=self.device)
        optimizer, scheduler = system.configure_optimizers()
        for epoch in range(max_epoch):
            dataloader = datamodule.train_dataloader()
            system.model.set_train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                system.on_train_batch_start(
                    batch, dataloader.dataset, epoch=epoch, global_step=self.global_step
                )
                loss = system.training_step(batch, self.global_step)
                loss["loss"].backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
                if self.global_step % cfg.log_every_n_steps == 0 or batch_idx == 0:
                    logger.info(f"Epoch {epoch}: {self.global_step}/{cfg.max_steps}")

                if self.global_step % cfg.val_check_interval == 0:
                    self.test(system, datamodule)

                if self.global_step == cfg.max_steps:
                    break
        self.export(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument("--resume", default=None, help="path to the weights")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    parser.add_argument("--exp_dir", default="./exp")

    args, extras = parser.parse_known_args()
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)
    config.trial_name = f"{config.tag}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.exp_dir = f"{args.exp_dir}/{config.name}"
    config.save_dir = f"{config.exp_dir}/{config.trial_name}/save"
    seed.set_seed(config.seed)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
