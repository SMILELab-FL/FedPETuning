"""Dry running Trainer"""

from abc import ABC
from utils import registry
from trainers import BaseTrainer
from run.centralized.client import CenClientTrainer


@registry.register_fl_algorithm("dry_run")
class DryTrainer(BaseTrainer, ABC):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_client(self):
        self.client_trainer = CenClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
        )
