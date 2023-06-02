"""Centralized Training for FedETuning"""

from abc import ABC
from trainers.BaseClient.base_client import BaseClientTrainer


class CenClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset):
        super().__init__(model, train_dataset, valid_dataset)
