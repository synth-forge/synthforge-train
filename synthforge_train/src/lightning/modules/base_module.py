from abc import ABC
from typing import Any

import torch
from lightning.pytorch import LightningModule


class BaseModule(LightningModule, ABC):
    _registry = {}

    def __init__(self, learning_rate: float = 1e-2, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = learning_rate

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        BaseModule._registry[cls.__name__] = cls

    @property
    def max_iter(self):
        return self.iters_per_epoch * self.trainer.max_epochs

    @property
    def iters_per_epoch(self):
        return len(self.trainer.train_dataloader)

    def current_iter(self, batch_idx):
        return self.current_epoch * self.iters_per_epoch + batch_idx

    def last_iter_in_epoch(self, batch_idx):
        return (self.current_iter(batch_idx) + 1) == self.iters_per_epoch

    @property
    def current_lr(self):
        return self.get_lr(self.optimizers())

    def get_current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def get_lr(self, optimizers):
        if not isinstance(optimizers, list):
            return self.get_current_lr(optimizers)
        else:
            return [self.get_current_lr(opt) for opt in optimizers]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        assert (
            getattr(self, "model", None) is not None
        ), f"Model is not defined for {self}"
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True,
        )

    def log_gradients_in_model(self, step, ignore_frozen=False):
        if not self.log_gradients:
            return
        none_grads = []
        for tag, value in self.named_parameters():
            if ignore_frozen and not value.requires_grad:
                continue
            if value.grad is not None:
                self.logger.experiment.add_histogram(
                    f"grad/{tag}", value.grad.cpu(), step
                )
            else:
                none_grads.append(tag)
        print(f"Grad None for: {tag}")

    def get_dl_name(self, dataloader_idx):
        if self.trainer.state.stage == "train":
            dls = self.trainer.datamodule.train_dataloader()
        elif self.trainer.state.stage in ["validate", "sanity_check"]:
            dls = self.trainer.datamodule.val_dataloader()
        elif self.trainer.state.stage == "test":
            dls = self.trainer.datamodule.test_dataloader()
        else:
            raise NotImplementedError

        if isinstance(dls, (list, dict)):
            dl_name = self.trainer.datamodule.dataloader_names[dataloader_idx]
            return f"{dl_name}/"
        return ""

    @property
    def log_path(self):
        log_path = f"{self.trainer.default_root_dir}/lightning_logs"
        log_path = f"{log_path}/version_{self.logger.version}/"
        return log_path
        