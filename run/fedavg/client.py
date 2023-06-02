"""federated average client"""

from abc import ABC
from trainers.BaseClient import BaseClientTrainer, BaseClientManager


class FedAvgClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset):
        super().__init__(model, train_dataset, valid_dataset)


class FedAvgClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)
