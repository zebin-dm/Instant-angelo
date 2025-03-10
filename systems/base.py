import torch
from systems.utils import parse_optimizer, parse_scheduler
from utils.utils import update_module_step
from utils.mixins import SaverMixin
from utils.misc import config_to_primitive
from torch.utils.tensorboard import SummaryWriter
from models.neus import NeuSModel


class BaseSystem(SaverMixin):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.prepare()
        self.writer: SummaryWriter = None
        self.device = device
        self.model = NeuSModel(self.config, self.device)

    def setup(self, writer):
        self.writer = writer
        self.model.to_device(self.device)

    def add_scalar(self, tag, value):
        if self.global_step % 100 != 0 and tag.startswith("train"):
            return
        self.writer.add_scalar(tag, value, global_step=self.global_step)

    def prepare(self):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError(
                    "Scalar specification only supports list, got", type(value)
                )
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
        return value

    def preprocess_data(self, batch, stage):
        raise NotImplementedError

    @torch.inference_mode()
    def on_train_batch_start(self, batch, dataset, epoch, global_step):
        self.current_epoch = epoch
        self.global_step = global_step
        self.preprocess_data(batch, "train", dataset)
        update_module_step(self.model, self.current_epoch, self.global_step)

    @torch.inference_mode()
    def on_test_batch_start(self, batch, dataset):
        self.preprocess_data(batch, "test", dataset)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = parse_optimizer(self.config.system.optimizer, self.model)
        scheduler = parse_scheduler(self.config.system.scheduler, optimizer)[
            "scheduler"
        ]
        return optimizer, scheduler
