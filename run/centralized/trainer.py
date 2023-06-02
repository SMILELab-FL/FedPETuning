"""centralized Trainer"""

import time
from abc import ABC
from trainers import BaseTrainer
from utils import registry, cen_metric_save
from run.centralized.client import CenClientTrainer


@registry.register_fl_algorithm("centralized")
class CentralizedTrainer(BaseTrainer, ABC):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_client(self):
        self.client_trainer = CenClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
        )

    def on_client_end(self):
        if self.training_config.do_predict:
            self.client_trainer.test_on_client(self.data.test_dataloader)
            cen_metric_save(self.client_trainer, self.training_config, self.logger)
