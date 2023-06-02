"""federated configs for FedETuning"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FederatedTrainingArguments:
    fl_algorithm: str = field(
        default="fedavg",
        metadata={"help": "The name of the federated learning algorithm"},
    )
    clients_num: int = field(
        default=10,
        metadata={"help": "The number of participant clients"},
    )
    alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "Non-IID shift and None denotes IID"},
    )
    partition_method: str = field(
        default=None,
        metadata={"help": "The partition methods"},
    )
    do_mimic: bool = field(
        default=True,
        metadata={"help": "Important! we only process once data processing in server if True"}
    )
    ip: str = field(
        default="127.0.0.1"
    )
    port: str = field(
        default="10001",
        metadata={"help": "The torch.distribute find this port to transmit stuff."}
    )
    rank: int = field(
        default=0, metadata={"help": "-1: centralized, 0: server, >0: client"}
    )
    world_size: int = field(
        default=None, metadata={"help": "The number of sub-server"}
    )
    ethernet: Optional[str] = field(
        default=None, metadata={"help": "not set"}
    )
    rounds: int = field(
        default=100, metadata={"help": "The number of training round"}
    )
    sample: float = field(
        default=0.1, metadata={"help": "The participant ratio in each training round"}
    )
    pson: bool = field(
        default=False, metadata={"help": "Whether to use personalized test(local) metric"}
    )
    test_rounds: bool = field(
        default=False, metadata={"help": "logging test(global) metric"}
    )
    log_test_len: int = field(
        default=10, metadata={"help": "logging test per communication rounds"}
    )
    _clients_num_per_sub_server: int = field(
        init=False, metadata={"help": "The number of clients in different works"}
    )

    def __post_init__(self):
        if self.alpha is None:
            # IID
            self.alpha = "inf"

        if not self.do_mimic:
            print("Please check whether federated device has its own data")

    @property
    def clients_num_per_sub_server(self):
        return int(self.clients_num / (self.world_size-1))

    @property
    def clients_id_list(self):
        if self.rank == -1:
            return [1]
        elif self.rank == 0:
            return [i for i in range(self.clients_num)]
        else:
            client_id_end = min(self.clients_num, self.rank * self.clients_num_per_sub_server)
            client_id_list = [
                i for i in range((self.rank - 1) * self.clients_num_per_sub_server, client_id_end)
            ]
            return client_id_list

    @property
    def clients_num_in_total(self):
        if self.rank == -1:
            # centralized
            return 1
        else:
            # federated
            return self.clients_num
