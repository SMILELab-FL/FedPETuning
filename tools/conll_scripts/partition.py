
import numpy as np
from fedlab.utils.dataset import DataPartitioner
from tools.partitions import label_skew_process


class NERDataPartition(DataPartitioner):
    def __init__(self, targets, num_clients, num_classes,
                 label_vocab, balance=True, partition="iid",
                 unbalance_sgm=0, num_shards=None,
                 dir_alpha=None, verbose=True, seed=42):

        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        # PRE-LOC
        self.label_vocab = label_vocab
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.num_classes = num_classes
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        # partition scheme check
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        # perform partition according to setting
        self.client_dict = self._perform_partition()
        # get sample number count for each client
        # self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        client_dict = label_skew_process(
            label_vocab=self.label_vocab, label_assignment=self.targets,
            client_num=self.num_clients, alpha=self.dir_alpha,
            data_length=len(self.targets)
        )

        return client_dict

    def __getitem__(self, index):
        """Obtain sample indices for client ``index``.

        Args:
            index (int): BaseClient ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        """Usually equals to number of clients."""
        return len(self.client_dict)


def get_partition_data(examples, label_list, num_clients, dir_alpha, partition):
    targets = [example["labels"] for example in examples]
    new_targets = []
    for target in targets:
        tags = filter(
            lambda x: x != "O",
            [token for token in target],
        )
        label_tags = set([t.split("-")[1] for t in tags])
        label_tags -= set(label_list)
        label = "-".join(sorted(label_tags))
        if label == "":
            label = "NULL"
        new_targets.append(label)
    label_vocab = set(new_targets)
    print(f"label vocab: {label_vocab}")
    num_classes = len(label_list)
    clients_partition_data = NERDataPartition(
        targets=new_targets, num_classes=num_classes, num_clients=num_clients,
        label_vocab=label_vocab, dir_alpha=dir_alpha, partition=partition, verbose=False
    )
    partition_data = {}
    for idx in range(len(clients_partition_data)):
        client_idxs = clients_partition_data[idx]
        partition_data[idx] = client_idxs
    return partition_data



