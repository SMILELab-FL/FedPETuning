"""federated average server"""


from abc import ABC

from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager


class FedAvgSyncServerHandler(BaseSyncServerHandler, ABC):
    def __init__(self, model, valid_data, test_data):
        super().__init__(model, valid_data, test_data)


class FedAvgServerManager(BaseServerManager, ABC):
    def __init__(self, network, handler):
        super().__init__(network, handler)
