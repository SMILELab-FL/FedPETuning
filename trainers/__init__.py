"""FedETuning's trainers registry in trainer.__init__.py -- IMPORTANT!"""

from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.trainer import FedAvgTrainer
from run.centralized.trainer import CenClientTrainer
from run.dry_run.trainer import DryTrainer

__all__ = [
    "BaseTrainer",
    "FedAvgTrainer",
    "DryTrainer",
    "CenClientTrainer"
]
